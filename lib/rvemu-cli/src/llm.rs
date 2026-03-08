/// OpenAI-compatible LLM client for generating RISC-V instructions from CPU state.

use crate::assembler;
use rvemu_core::cpu::Cpu;
use serde::{Deserialize, Serialize};

/// A record of a previously executed instruction.
pub struct InstructionRecord {
    pub pc: u64,
    pub inst_hex: u64,
    pub description: String,
}

/// The LLM executor that manages OpenAI-compatible API communication and instruction history.
pub struct LlmExecutor {
    client: reqwest::blocking::Client,
    pub model: String,
    pub api_base_url: String,
    pub api_key: String,
    pub goal_memory: Option<Vec<u8>>,
    pub history: Vec<InstructionRecord>,
}

impl LlmExecutor {
    pub fn new(
        model: String,
        api_base_url: String,
        api_key: String,
        goal_memory: Option<Vec<u8>>,
    ) -> Result<Self, String> {
        let client = reqwest::blocking::Client::builder()
            // Avoid macOS system proxy autodiscovery path, which can panic on
            // some host environments.
            .no_proxy()
            .build()
            .map_err(|e| format!("failed to initialize HTTP client: {}", e))?;

        Ok(Self {
            client,
            model,
            api_base_url,
            api_key,
            goal_memory,
            history: Vec::new(),
        })
    }

    /// Build a prompt from the current CPU state and instruction history.
    pub fn build_prompt(&self, cpu: &Cpu) -> String {
        let mut prompt = String::new();

        prompt.push_str(
            "You are a RISC-V CPU simulator. Given the current CPU state, generate the next \
             single RISC-V instruction to execute. You MUST respond with ONLY the instruction, \
             nothing else. The instruction must be a single RISC-V assembly mnemonic \
             (e.g., addi a0, x0, 10).\n\n\
             Do NOT include any explanation, comments, or extra text. Just the instruction.\n\n",
        );

        // Register state
        prompt.push_str("=== Current CPU State ===\n");
        prompt.push_str(&format!("PC: {:#x}\n", cpu.pc));
        prompt.push_str(&format!("{}\n", cpu.xregs));

        // Memory state
        let dram = cpu.bus.dram();
        let dram_size = dram.size();
        prompt.push_str(&format!(
            "DRAM base: 0x80000000, DRAM size: {} bytes, valid DRAM range: [0x80000000, {:#x}]\n",
            dram_size,
            0x8000_0000u64 + dram_size.saturating_sub(1)
        ));
        if dram_size <= 1024 {
            // Small DRAM: dump everything
            prompt.push_str(&format!("{}\n", dram));
        } else {
            // Large DRAM: show a window around PC
            prompt.push_str(&format!("Memory (showing window around PC, total {} bytes):\n", dram_size));
            // Show 256 bytes around PC
            let dram_base: u64 = 0x80000000;
            let pc_offset = cpu.pc.saturating_sub(dram_base);
            let start = pc_offset.saturating_sub(64);
            let end = std::cmp::min(pc_offset + 192, dram_size);
            for offset in (start..end).step_by(16) {
                let end_byte = std::cmp::min(offset + 16, end);
                prompt.push_str(&format!("{:08x}: ", dram_base + offset));
                for i in offset..end_byte {
                    if i > offset && i % 2 == 0 {
                        prompt.push(' ');
                    }
                    prompt.push_str(&format!("{:02x}", dram.dram[i as usize]));
                }
                prompt.push('\n');
            }
        }

        if let Some(goal) = &self.goal_memory {
            prompt.push_str(
                "\n=== Goal Memory ===\n\
                 The bytes below are the desired DRAM contents starting at 0x80000000.\n\
                 Bytes not specified in this goal are expected to remain zero.\n",
            );
            let preview_len = std::cmp::min(goal.len(), 256);
            prompt.push_str(&format!(
                "Goal file bytes: {} (showing first {} bytes)\n",
                goal.len(),
                preview_len
            ));
            for offset in (0..preview_len).step_by(16) {
                let end = std::cmp::min(offset + 16, preview_len);
                prompt.push_str(&format!("{:08x}: ", 0x8000_0000u64 + offset as u64));
                for i in offset..end {
                    if i > offset && i % 2 == 0 {
                        prompt.push(' ');
                    }
                    prompt.push_str(&format!("{:02x}", goal[i]));
                }
                prompt.push('\n');
            }
        }

        // Instruction history (last 100)
        if !self.history.is_empty() {
            prompt.push_str("\n=== Last Instructions Executed ===\n");
            let start = if self.history.len() > 100 {
                self.history.len() - 100
            } else {
                0
            };
            for record in &self.history[start..] {
                prompt.push_str(&format!(
                    "PC={:#x} inst={:#010x} {}\n",
                    record.pc, record.inst_hex, record.description
                ));
            }
        }

        prompt.push_str("\nNext instruction:\n");
        prompt
    }

    /// Call an OpenAI-compatible Chat Completions API to generate a response.
    pub fn call_api(&self, prompt: &str) -> Result<String, String> {
        let chat_url = format!(
            "{}/v1/chat/completions",
            self.api_base_url.trim_end_matches('/')
        );
        let chat_body = ChatCompletionsRequest {
            model: self.model.clone(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: 0.0,
        };

        let response = self
            .client
            .post(&chat_url)
            .bearer_auth(&self.api_key)
            .json(&chat_body)
            .send()
            .map_err(|e| format!("API request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!(
                "API returned status {}: {}",
                response.status(),
                response.text().unwrap_or_default()
            ));
        }

        let json: ChatCompletionsResponse = response
            .json()
            .map_err(|e| format!("Failed to parse API response: {}", e))?;

        json.choices
            .first()
            .and_then(|choice| choice.message.content.as_deref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "No text content in API response choices[0].message.content".to_string())
    }

    /// Parse the LLM response into a u64 instruction.
    /// Only assembly responses are accepted.
    pub fn parse_response(&self, response: &str) -> Result<u64, String> {
        let response = response.trim();
        if response.starts_with("0x") || response.starts_with("0X") {
            return Err(
                "hex machine-word responses are not supported; return assembly mnemonic"
                    .to_string(),
            );
        }

        let inst = assembler::assemble(response)?;
        Ok(inst as u64)
    }

    /// Record an executed instruction in history, keeping only the last 100.
    pub fn record_instruction(&mut self, pc: u64, inst_hex: u64, description: String) {
        self.history.push(InstructionRecord {
            pc,
            inst_hex,
            description,
        });
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }
}

#[derive(Serialize)]
struct ChatCompletionsRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionsResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::LlmExecutor;

    fn make_executor() -> LlmExecutor {
        LlmExecutor::new(
            "dummy".to_string(),
            "https://api.openai.com".to_string(),
            "dummy".to_string(),
            None,
        )
        .expect("executor should initialize")
    }

    #[test]
    fn parse_response_rejects_hex_machine_word() {
        let ex = make_executor();
        let err = ex
            .parse_response("0x02a00293")
            .expect_err("hex instruction should not parse");
        assert!(err.contains("not supported"));
    }

    #[test]
    fn parse_response_prefers_assembly_when_line_contains_hex_immediate() {
        let ex = make_executor();
        let inst = ex
            .parse_response("lui t2, 0x80000")
            .expect("assembly should parse");
        assert_eq!(inst, 0x8000_03b7);
    }
}

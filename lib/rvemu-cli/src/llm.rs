/// Ollama LLM client for generating RISC-V instructions from CPU state.

use crate::assembler;
use rvemu_core::cpu::Cpu;

/// A record of a previously executed instruction.
pub struct InstructionRecord {
    pub pc: u64,
    pub inst_hex: u64,
    pub description: String,
}

/// The LLM executor that manages Ollama communication and instruction history.
pub struct LlmExecutor {
    client: reqwest::blocking::Client,
    pub model: String,
    pub ollama_url: String,
    pub history: Vec<InstructionRecord>,
}

impl LlmExecutor {
    pub fn new(model: String, ollama_url: String) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            model,
            ollama_url,
            history: Vec::new(),
        }
    }

    /// Build a prompt from the current CPU state and instruction history.
    pub fn build_prompt(&self, cpu: &Cpu) -> String {
        let mut prompt = String::new();

        prompt.push_str(
            "You are a RISC-V CPU simulator. Given the current CPU state, generate the next \
             single RISC-V instruction to execute. You MUST respond with ONLY the instruction, \
             nothing else. The instruction can be either:\n\
             1. A hex-encoded 32-bit instruction (e.g., 0x00a00513)\n\
             2. A RISC-V assembly mnemonic (e.g., addi a0, x0, 10)\n\n\
             Do NOT include any explanation, comments, or extra text. Just the instruction.\n\n",
        );

        // Register state
        prompt.push_str("=== Current CPU State ===\n");
        prompt.push_str(&format!("PC: {:#x}\n", cpu.pc));
        prompt.push_str(&format!("{}\n", cpu.xregs));

        // Memory state
        let dram = cpu.bus.dram();
        let dram_size = dram.size();
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

    /// Call the Ollama API to generate a response.
    pub fn call_ollama(&self, prompt: &str) -> Result<String, String> {
        let url = format!("{}/api/generate", self.ollama_url);
        let body = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false
        });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| format!("Ollama request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!(
                "Ollama returned status {}: {}",
                response.status(),
                response.text().unwrap_or_default()
            ));
        }

        let json: serde_json::Value = response
            .json()
            .map_err(|e| format!("Failed to parse Ollama response: {}", e))?;

        json["response"]
            .as_str()
            .map(|s| s.trim().to_string())
            .ok_or_else(|| "No 'response' field in Ollama output".to_string())
    }

    /// Parse the LLM response into a u64 instruction.
    /// Tries hex extraction first, then falls back to assembly parsing.
    pub fn parse_response(&self, response: &str) -> Result<u64, String> {
        let response = response.trim();

        // Try to find a hex instruction (0x prefix)
        for token in response.split_whitespace() {
            let token = token.trim_end_matches(|c: char| !c.is_ascii_hexdigit() && c != 'x' && c != 'X');
            if let Some(hex_str) = token.strip_prefix("0x").or_else(|| token.strip_prefix("0X")) {
                if let Ok(val) = u64::from_str_radix(hex_str, 16) {
                    if val <= 0xFFFFFFFF {
                        return Ok(val);
                    }
                }
            }
        }

        // Fall back to assembler
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

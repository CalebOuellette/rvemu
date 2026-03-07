use clap::{App, Arg};
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::iter::FromIterator;

use rvemu_core::bus::DRAM_BASE;
use rvemu_core::cpu::Cpu;
use rvemu_core::dram::DRAM_SIZE;
use rvemu_core::emulator::Emulator;
use rvemu_core::exception::Trap;

mod assembler;
mod llm;

/// Output current registers to the console.
fn dump_registers(cpu: &Cpu) {
    println!("-------------------------------------------------------------------------------------------");
    println!("{}", cpu.xregs);
    println!("-------------------------------------------------------------------------------------------");
    println!("{}", cpu.fregs);
    println!("-------------------------------------------------------------------------------------------");
    println!("{}", cpu.state);
    println!("-------------------------------------------------------------------------------------------");
    println!("pc: {:#x}", cpu.pc);
}

/// Output the count of each instruction executed.
fn dump_count(cpu: &Cpu) {
    if cpu.is_count {
        println!("===========================================================================================");
        let mut sorted_counter = Vec::from_iter(&cpu.inst_counter);
        sorted_counter.sort_by(|&(_, a), &(_, b)| b.cmp(&a));
        for (inst, count) in sorted_counter.iter() {
            println!("{}, {}", inst, count);
        }
        println!("===========================================================================================");
    }
}

/// Run the emulator in LLM mode: instead of fetching instructions from DRAM,
/// ask an OpenAI-compatible LLM for the next instruction based on current CPU state.
fn llm_start(
    emu: &mut Emulator,
    executor: &mut llm::LlmExecutor,
    max_instructions: Option<u64>,
) {
    let max_retries = 3;
    let mut executed_instructions = 0u64;

    loop {
        // Run a cycle on peripheral devices.
        emu.cpu.devices_increment();

        // Take an interrupt.
        if let Some(interrupt) = emu.cpu.check_pending_interrupt() {
            interrupt.take_trap(&mut emu.cpu);
        }

        // Build prompt and call LLM
        let prompt = executor.build_prompt(&emu.cpu);
        let current_pc = emu.cpu.pc;

        println!("[LLM] === Prompt ===\n{}\n[LLM] === End Prompt ===", prompt);

        let mut response = None;
        for attempt in 0..max_retries {
            match executor.call_api(&prompt) {
                Ok(resp) => {
                    response = Some(resp);
                    break;
                }
                Err(e) => {
                    eprintln!(
                        "[LLM] Attempt {}/{} failed: {}",
                        attempt + 1,
                        max_retries,
                        e
                    );
                    if attempt + 1 == max_retries {
                        eprintln!("[LLM] All retries exhausted. Halting.");
                        return;
                    }
                }
            }
        }

        let response_text = response.unwrap();
        println!("[LLM] Response: {}", response_text);

        // Parse response into instruction
        let inst = match executor.parse_response(&response_text) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("[LLM] Failed to parse instruction: {}", e);
                eprintln!("[LLM] Raw response was: {}", response_text);
                continue;
            }
        };

        println!("[LLM] PC={:#x} Executing instruction: {:#010x}", current_pc, inst);

        // Determine if compressed (bits [1:0] != 0b11) or general
        let trap = if inst & 0b11 != 0b11 {
            // Compressed instruction
            match emu.cpu.execute_compressed(inst) {
                Ok(_) => {
                    emu.cpu.pc += 2;
                    Trap::Requested
                }
                Err(exception) => exception.take_trap(&mut emu.cpu),
            }
        } else {
            // General (32-bit) instruction
            match emu.cpu.execute_general(inst) {
                Ok(_) => {
                    emu.cpu.pc += 4;
                    Trap::Requested
                }
                Err(exception) => exception.take_trap(&mut emu.cpu),
            }
        };

        // Record instruction
        executor.record_instruction(current_pc, inst, response_text.clone());
        executed_instructions += 1;

        // Print state after execution
        println!(
            "[LLM] PC after: {:#x}",
            emu.cpu.pc
        );

        match trap {
            Trap::Fatal => {
                println!("[LLM] Fatal trap at PC={:#x}", emu.cpu.pc);
                return;
            }
            _ => {}
        }

        if let Some(limit) = max_instructions {
            if executed_instructions >= limit {
                println!(
                    "[LLM] Reached instruction limit ({}). Halting.",
                    limit
                );
                return;
            }
        }
    }
}

/// Main function of RISC-V emulator for the CLI version.
fn main() -> io::Result<()> {
    // Load environment variables from .env in the current directory or parents.
    let _ = dotenvy::dotenv();

    let matches = App::new("rvemu: RISC-V emulator")
        .version("0.0.1")
        .author("Asami Doi <@d0iasm>")
        .arg(
            Arg::with_name("kernel")
                .short("k")
                .long("kernel")
                .takes_value(true)
                .required(true)
                .help("A kernel ELF image without headers"),
        )
        .arg(
            Arg::with_name("file")
                .short("f")
                .long("file")
                .takes_value(true)
                .help("A raw disk image"),
        )
        .arg(
            Arg::with_name("debug")
                .short("d")
                .long("debug")
                .help("Enables to output debug messages"),
        )
        .arg(
            Arg::with_name("count")
                .short("c")
                .long("count")
                .help("Enables to count each instruction executed"),
        )
        .arg(
            Arg::with_name("memory-size")
                .short("m")
                .long("memory-size")
                .takes_value(true)
                .help("DRAM size in bytes (default: 1 GiB)"),
        )
        .arg(
            Arg::with_name("llm")
                .long("llm")
                .help("Enable LLM-driven instruction execution via an OpenAI-compatible API"),
        )
        .arg(
            Arg::with_name("model")
                .long("model")
                .takes_value(true)
                .help("Model name for the OpenAI-compatible API (overrides OPENAI_MODEL)"),
        )
        .arg(
            Arg::with_name("llm-max-instructions")
                .long("llm-max-instructions")
                .takes_value(true)
                .help("Maximum number of LLM-generated instructions to execute before halting"),
        )
        .get_matches();

    let mut kernel_file = File::open(
        &matches
            .value_of("kernel")
            .expect("failed to get a kernel file from a command option"),
    )?;
    let mut kernel_data = Vec::new();
    kernel_file.read_to_end(&mut kernel_data)?;

    let mut img_data = Vec::new();
    if let Some(img_file) = matches.value_of("file") {
        File::open(img_file)?.read_to_end(&mut img_data)?;
    }

    let dram_size = matches
        .value_of("memory-size")
        .map(|s| s.parse::<u64>().expect("memory-size must be a number"))
        .unwrap_or(DRAM_SIZE);

    let mut emu = Emulator::with_dram_size(dram_size);

    emu.initialize_dram(kernel_data);
    emu.initialize_disk(img_data);
    emu.initialize_pc(DRAM_BASE);

    if matches.occurrences_of("debug") == 1 {
        emu.is_debug = true;
    }

    if matches.occurrences_of("count") == 1 {
        emu.cpu.is_count = true;
    }

    if matches.is_present("llm") {
        let llm_max_instructions = matches
            .value_of("llm-max-instructions")
            .map(|s| {
                s.parse::<u64>()
                    .expect("llm-max-instructions must be a positive integer")
            });
        let model = matches
            .value_of("model")
            .map(|s| s.to_string())
            .or_else(|| std::env::var("OPENAI_MODEL").ok())
            .or_else(|| std::env::var("OPENAI_API_MODEL").ok())
            .unwrap_or_else(|| "gpt-5.3-codex".to_string());
        let api_base_url = std::env::var("OPENAI_API_BASE_URL")
            .or_else(|_| std::env::var("OPENAI_BASE_URL"))
            .or_else(|_| std::env::var("OPENAI_URL"))
            .unwrap_or_else(|_| "https://api.openai.com".to_string());
        let api_key = std::env::var("OPENAI_API_KEY")
            .or_else(|_| std::env::var("OPENAI_KEY"))
            .expect(
                "OPENAI_API_KEY is required in --llm mode (fallback: OPENAI_KEY)",
            );
        println!(
            "[LLM] Starting LLM-driven execution with model '{}' at {}",
            model, api_base_url
        );
        let mut executor = llm::LlmExecutor::new(model, api_base_url, api_key);
        llm_start(&mut emu, &mut executor, llm_max_instructions);
    } else {
        emu.start();
    }

    dump_registers(&emu.cpu);
    dump_count(&emu.cpu);

    Ok(())
}

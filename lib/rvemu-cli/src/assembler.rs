/// A basic RISC-V assembler that converts text mnemonics to 32-bit instruction words.

fn parse_register(name: &str) -> Result<u32, String> {
    let name = name.trim().trim_end_matches(',');
    // Try x0-x31 form
    if let Some(num_str) = name.strip_prefix('x') {
        if let Ok(n) = num_str.parse::<u32>() {
            if n <= 31 {
                return Ok(n);
            }
        }
    }
    // ABI names
    match name {
        "zero" => Ok(0),
        "ra" => Ok(1),
        "sp" => Ok(2),
        "gp" => Ok(3),
        "tp" => Ok(4),
        "t0" => Ok(5),
        "t1" => Ok(6),
        "t2" => Ok(7),
        "s0" | "fp" => Ok(8),
        "s1" => Ok(9),
        "a0" => Ok(10),
        "a1" => Ok(11),
        "a2" => Ok(12),
        "a3" => Ok(13),
        "a4" => Ok(14),
        "a5" => Ok(15),
        "a6" => Ok(16),
        "a7" => Ok(17),
        "s2" => Ok(18),
        "s3" => Ok(19),
        "s4" => Ok(20),
        "s5" => Ok(21),
        "s6" => Ok(22),
        "s7" => Ok(23),
        "s8" => Ok(24),
        "s9" => Ok(25),
        "s10" => Ok(26),
        "s11" => Ok(27),
        "t3" => Ok(28),
        "t4" => Ok(29),
        "t5" => Ok(30),
        "t6" => Ok(31),
        _ => Err(format!("unknown register: {}", name)),
    }
}

fn parse_immediate(s: &str) -> Result<i64, String> {
    let s = s.trim().trim_end_matches(',');
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).map_err(|e| format!("bad hex immediate '{}': {}", s, e))
    } else if let Some(hex) = s.strip_prefix("-0x").or_else(|| s.strip_prefix("-0X")) {
        i64::from_str_radix(hex, 16)
            .map(|v| -v)
            .map_err(|e| format!("bad hex immediate '{}': {}", s, e))
    } else {
        s.parse::<i64>()
            .map_err(|e| format!("bad immediate '{}': {}", s, e))
    }
}

/// Parse `offset(rs1)` syntax used by load/store instructions.
/// Returns (offset, rs1_register_number).
fn parse_mem_operand(s: &str) -> Result<(i64, u32), String> {
    let s = s.trim();
    let open = s
        .find('(')
        .ok_or_else(|| format!("expected offset(reg) syntax, got '{}'", s))?;
    let close = s
        .find(')')
        .ok_or_else(|| format!("expected offset(reg) syntax, got '{}'", s))?;
    let offset = parse_immediate(&s[..open])?;
    let reg = parse_register(&s[open + 1..close])?;
    Ok((offset, reg))
}

fn encode_r(funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

fn encode_i(imm: i64, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    let imm = (imm as u32) & 0xFFF;
    (imm << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

fn encode_s(imm: i64, rs2: u32, rs1: u32, funct3: u32, opcode: u32) -> u32 {
    let imm = (imm as u32) & 0xFFF;
    let imm_11_5 = (imm >> 5) & 0x7F;
    let imm_4_0 = imm & 0x1F;
    (imm_11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_4_0 << 7) | opcode
}

fn encode_b(imm: i64, rs2: u32, rs1: u32, funct3: u32, opcode: u32) -> u32 {
    let imm = (imm as u32) & 0x1FFF;
    let bit12 = (imm >> 12) & 1;
    let bits10_5 = (imm >> 5) & 0x3F;
    let bits4_1 = (imm >> 1) & 0xF;
    let bit11 = (imm >> 11) & 1;
    (bit12 << 31)
        | (bits10_5 << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (bits4_1 << 8)
        | (bit11 << 7)
        | opcode
}

fn encode_u(imm: i64, rd: u32, opcode: u32) -> u32 {
    let imm = (imm as u32) & 0xFFFFF000;
    imm | (rd << 7) | opcode
}

fn encode_j(imm: i64, rd: u32, opcode: u32) -> u32 {
    let imm = (imm as u32) & 0x1FFFFF;
    let bit20 = (imm >> 20) & 1;
    let bits10_1 = (imm >> 1) & 0x3FF;
    let bit11 = (imm >> 11) & 1;
    let bits19_12 = (imm >> 12) & 0xFF;
    (bit20 << 31) | (bits10_1 << 21) | (bit11 << 20) | (bits19_12 << 12) | (rd << 7) | opcode
}

/// Assemble a single RISC-V instruction from text to a 32-bit machine word.
pub fn assemble(line: &str) -> Result<u32, String> {
    let line = line.trim();
    if line.is_empty() {
        return Err("empty instruction".to_string());
    }

    // Split into mnemonic and operands
    let parts: Vec<&str> = line.splitn(2, char::is_whitespace).collect();
    let mnemonic = parts[0].to_lowercase();
    let operands_str = if parts.len() > 1 { parts[1].trim() } else { "" };

    // Helper to split comma-separated operands
    let ops: Vec<&str> = if operands_str.is_empty() {
        vec![]
    } else {
        operands_str.split(',').map(|s| s.trim()).collect()
    };

    match mnemonic.as_str() {
        // Special instructions
        "nop" => Ok(encode_i(0, 0, 0, 0, 0x13)), // addi x0, x0, 0
        "ecall" => Ok(0x00000073),
        "ebreak" => Ok(0x00100073),

        // R-type: op rd, rs1, rs2
        "add" | "sub" | "and" | "or" | "xor" | "sll" | "srl" | "sra" | "slt" | "sltu" => {
            if ops.len() != 3 {
                return Err(format!("{} requires 3 operands", mnemonic));
            }
            let rd = parse_register(ops[0])?;
            let rs1 = parse_register(ops[1])?;
            let rs2 = parse_register(ops[2])?;
            let (funct7, funct3) = match mnemonic.as_str() {
                "add" => (0x00, 0x0),
                "sub" => (0x20, 0x0),
                "sll" => (0x00, 0x1),
                "slt" => (0x00, 0x2),
                "sltu" => (0x00, 0x3),
                "xor" => (0x00, 0x4),
                "srl" => (0x00, 0x5),
                "sra" => (0x20, 0x5),
                "or" => (0x00, 0x6),
                "and" => (0x00, 0x7),
                _ => unreachable!(),
            };
            Ok(encode_r(funct7, rs2, rs1, funct3, rd, 0x33))
        }

        // I-type arithmetic: op rd, rs1, imm
        "addi" | "slti" | "sltiu" | "andi" | "ori" | "xori" | "slli" | "srli" | "srai" => {
            if ops.len() != 3 {
                return Err(format!("{} requires 3 operands", mnemonic));
            }
            let rd = parse_register(ops[0])?;
            let rs1 = parse_register(ops[1])?;
            let imm = parse_immediate(ops[2])?;
            let funct3 = match mnemonic.as_str() {
                "addi" => 0x0,
                "slti" => 0x2,
                "sltiu" => 0x3,
                "xori" => 0x4,
                "ori" => 0x6,
                "andi" => 0x7,
                "slli" => 0x1,
                "srli" => 0x5,
                "srai" => 0x5,
                _ => unreachable!(),
            };
            match mnemonic.as_str() {
                "slli" => Ok(encode_i(imm & 0x3F, rs1, funct3, rd, 0x13)),
                "srli" => Ok(encode_i(imm & 0x3F, rs1, funct3, rd, 0x13)),
                "srai" => {
                    let shamt = (imm & 0x3F) | 0x400; // set bit 10 for srai
                    Ok(encode_i(shamt, rs1, funct3, rd, 0x13))
                }
                _ => Ok(encode_i(imm, rs1, funct3, rd, 0x13)),
            }
        }

        // I-type loads: op rd, offset(rs1)
        "lb" | "lh" | "lw" | "ld" | "lbu" | "lhu" | "lwu" => {
            if ops.len() != 2 {
                return Err(format!("{} requires 2 operands (rd, offset(rs1))", mnemonic));
            }
            let rd = parse_register(ops[0])?;
            let (offset, rs1) = parse_mem_operand(ops[1])?;
            let funct3 = match mnemonic.as_str() {
                "lb" => 0x0,
                "lh" => 0x1,
                "lw" => 0x2,
                "ld" => 0x3,
                "lbu" => 0x4,
                "lhu" => 0x5,
                "lwu" => 0x6,
                _ => unreachable!(),
            };
            Ok(encode_i(offset, rs1, funct3, rd, 0x03))
        }

        // S-type stores: op rs2, offset(rs1)
        "sb" | "sh" | "sw" | "sd" => {
            if ops.len() != 2 {
                return Err(format!(
                    "{} requires 2 operands (rs2, offset(rs1))",
                    mnemonic
                ));
            }
            let rs2 = parse_register(ops[0])?;
            let (offset, rs1) = parse_mem_operand(ops[1])?;
            let funct3 = match mnemonic.as_str() {
                "sb" => 0x0,
                "sh" => 0x1,
                "sw" => 0x2,
                "sd" => 0x3,
                _ => unreachable!(),
            };
            Ok(encode_s(offset, rs2, rs1, funct3, 0x23))
        }

        // B-type branches: op rs1, rs2, offset
        "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu" => {
            if ops.len() != 3 {
                return Err(format!("{} requires 3 operands", mnemonic));
            }
            let rs1 = parse_register(ops[0])?;
            let rs2 = parse_register(ops[1])?;
            let imm = parse_immediate(ops[2])?;
            let funct3 = match mnemonic.as_str() {
                "beq" => 0x0,
                "bne" => 0x1,
                "blt" => 0x4,
                "bge" => 0x5,
                "bltu" => 0x6,
                "bgeu" => 0x7,
                _ => unreachable!(),
            };
            Ok(encode_b(imm, rs2, rs1, funct3, 0x63))
        }

        // U-type: op rd, imm
        "lui" | "auipc" => {
            if ops.len() != 2 {
                return Err(format!("{} requires 2 operands", mnemonic));
            }
            let rd = parse_register(ops[0])?;
            let imm = parse_immediate(ops[1])?;
            // The immediate for lui/auipc is the upper 20 bits.
            // If user provides the raw upper value (e.g., 0x12345), shift it up.
            // Convention: value is placed in bits [31:12].
            let imm_val = (imm as u32) << 12;
            let opcode = match mnemonic.as_str() {
                "lui" => 0x37,
                "auipc" => 0x17,
                _ => unreachable!(),
            };
            Ok(encode_u(imm_val as i64, rd, opcode))
        }

        // J-type: jal rd, offset
        "jal" => {
            if ops.len() == 1 {
                // jal offset  (rd defaults to ra)
                let imm = parse_immediate(ops[0])?;
                Ok(encode_j(imm, 1, 0x6F))
            } else if ops.len() == 2 {
                let rd = parse_register(ops[0])?;
                let imm = parse_immediate(ops[1])?;
                Ok(encode_j(imm, rd, 0x6F))
            } else {
                Err("jal requires 1 or 2 operands".to_string())
            }
        }

        // I-type jump: jalr rd, rs1, offset
        "jalr" => {
            if ops.len() == 2 {
                // jalr rd, offset(rs1) or jalr rd, rs1
                let rd = parse_register(ops[0])?;
                if ops[1].contains('(') {
                    let (offset, rs1) = parse_mem_operand(ops[1])?;
                    Ok(encode_i(offset, rs1, 0x0, rd, 0x67))
                } else {
                    let rs1 = parse_register(ops[1])?;
                    Ok(encode_i(0, rs1, 0x0, rd, 0x67))
                }
            } else if ops.len() == 3 {
                let rd = parse_register(ops[0])?;
                let rs1 = parse_register(ops[1])?;
                let offset = parse_immediate(ops[2])?;
                Ok(encode_i(offset, rs1, 0x0, rd, 0x67))
            } else {
                Err("jalr requires 2 or 3 operands".to_string())
            }
        }

        // RV64I word-sized ops
        "addw" | "subw" | "sllw" | "srlw" | "sraw" => {
            if ops.len() != 3 {
                return Err(format!("{} requires 3 operands", mnemonic));
            }
            let rd = parse_register(ops[0])?;
            let rs1 = parse_register(ops[1])?;
            let rs2 = parse_register(ops[2])?;
            let (funct7, funct3) = match mnemonic.as_str() {
                "addw" => (0x00, 0x0),
                "subw" => (0x20, 0x0),
                "sllw" => (0x00, 0x1),
                "srlw" => (0x00, 0x5),
                "sraw" => (0x20, 0x5),
                _ => unreachable!(),
            };
            Ok(encode_r(funct7, rs2, rs1, funct3, rd, 0x3B))
        }

        "addiw" | "slliw" | "srliw" | "sraiw" => {
            if ops.len() != 3 {
                return Err(format!("{} requires 3 operands", mnemonic));
            }
            let rd = parse_register(ops[0])?;
            let rs1 = parse_register(ops[1])?;
            let imm = parse_immediate(ops[2])?;
            let funct3 = match mnemonic.as_str() {
                "addiw" => 0x0,
                "slliw" => 0x1,
                "srliw" => 0x5,
                "sraiw" => 0x5,
                _ => unreachable!(),
            };
            match mnemonic.as_str() {
                "slliw" => Ok(encode_i(imm & 0x1F, rs1, funct3, rd, 0x1B)),
                "srliw" => Ok(encode_i(imm & 0x1F, rs1, funct3, rd, 0x1B)),
                "sraiw" => {
                    let shamt = (imm & 0x1F) | 0x400;
                    Ok(encode_i(shamt, rs1, funct3, rd, 0x1B))
                }
                _ => Ok(encode_i(imm, rs1, funct3, rd, 0x1B)),
            }
        }

        // Pseudo-instructions
        "li" => {
            // li rd, imm -> addi rd, x0, imm (only works for small immediates)
            if ops.len() != 2 {
                return Err("li requires 2 operands".to_string());
            }
            let rd = parse_register(ops[0])?;
            let imm = parse_immediate(ops[1])?;
            Ok(encode_i(imm, 0, 0x0, rd, 0x13))
        }

        "mv" => {
            // mv rd, rs1 -> addi rd, rs1, 0
            if ops.len() != 2 {
                return Err("mv requires 2 operands".to_string());
            }
            let rd = parse_register(ops[0])?;
            let rs1 = parse_register(ops[1])?;
            Ok(encode_i(0, rs1, 0x0, rd, 0x13))
        }

        "ret" => {
            // ret -> jalr x0, x1, 0
            Ok(encode_i(0, 1, 0x0, 0, 0x67))
        }

        "j" => {
            // j offset -> jal x0, offset
            if ops.len() != 1 {
                return Err("j requires 1 operand".to_string());
            }
            let imm = parse_immediate(ops[0])?;
            Ok(encode_j(imm, 0, 0x6F))
        }

        _ => Err(format!("unsupported instruction: {}", mnemonic)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addi() {
        // addi a0, x0, 10 should be 0x00a00513
        let inst = assemble("addi a0, x0, 10").unwrap();
        assert_eq!(inst, 0x00a00513);
    }

    #[test]
    fn test_add() {
        // add x1, x2, x3
        let inst = assemble("add x1, x2, x3").unwrap();
        assert_eq!(inst, 0x003100b3);
    }

    #[test]
    fn test_nop() {
        let inst = assemble("nop").unwrap();
        assert_eq!(inst, 0x00000013);
    }

    #[test]
    fn test_ecall() {
        let inst = assemble("ecall").unwrap();
        assert_eq!(inst, 0x00000073);
    }

    #[test]
    fn test_sw() {
        // sw x5, 8(x2) -> offset=8, rs2=5, rs1=2, funct3=2, opcode=0x23
        let inst = assemble("sw x5, 8(x2)").unwrap();
        assert_eq!(inst, 0x00512423);
    }

    #[test]
    fn test_lw() {
        // lw x5, 8(x2)
        let inst = assemble("lw x5, 8(x2)").unwrap();
        assert_eq!(inst, 0x00812283);
    }

    #[test]
    fn test_beq() {
        // beq x1, x2, 8
        let inst = assemble("beq x1, x2, 8").unwrap();
        assert_eq!(inst, 0x00208463);
    }

    #[test]
    fn test_lui() {
        // lui x1, 0x12345 -> upper 20 bits = 0x12345, rd=1
        let inst = assemble("lui x1, 0x12345").unwrap();
        assert_eq!(inst, 0x123450b7);
    }

    #[test]
    fn test_li() {
        // li a0, 42 -> addi a0, x0, 42
        let inst = assemble("li a0, 42").unwrap();
        assert_eq!(inst, 0x02a00513);
    }

    #[test]
    fn test_ret() {
        // ret -> jalr x0, x1, 0 = 0x00008067
        let inst = assemble("ret").unwrap();
        assert_eq!(inst, 0x00008067);
    }
}

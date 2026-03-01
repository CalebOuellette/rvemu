//! The memory module contains the memory structure and implementation to read/write the memory.

use std::fmt;

use crate::bus::DRAM_BASE;
use crate::cpu::{BYTE, DOUBLEWORD, HALFWORD, WORD};
use crate::exception::Exception;

/// Default memory size (64 bytes).
pub const DRAM_SIZE: u64 = 64;

/// The memory used by the emulator.
#[derive(Debug)]
pub struct Dram {
    pub dram: Vec<u8>,
    code_size: u64,
    size: u64,
}

impl Dram {
    /// Create a new memory object with default memory size.
    pub fn new() -> Self {
        Self::with_size(DRAM_SIZE)
    }

    /// Create a new memory object with a custom size in bytes.
    pub fn with_size(size: u64) -> Self {
        Self {
            dram: vec![0; size as usize],
            code_size: 0,
            size,
        }
    }

    /// Return the size of the DRAM in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Set the binary in the memory.
    pub fn initialize(&mut self, binary: Vec<u8>) {
        self.code_size = binary.len() as u64;
        self.dram.splice(..binary.len(), binary.iter().cloned());
    }

    /// Load `size`-bit data from the memory.
    pub fn read(&self, addr: u64, size: u8) -> Result<u64, Exception> {
        match size {
            BYTE => Ok(self.read8(addr)),
            HALFWORD => Ok(self.read16(addr)),
            WORD => Ok(self.read32(addr)),
            DOUBLEWORD => Ok(self.read64(addr)),
            _ => return Err(Exception::LoadAccessFault),
        }
    }

    /// Store `size`-bit data to the memory.
    pub fn write(&mut self, addr: u64, value: u64, size: u8) -> Result<(), Exception> {
        match size {
            BYTE => self.write8(addr, value),
            HALFWORD => self.write16(addr, value),
            WORD => self.write32(addr, value),
            DOUBLEWORD => self.write64(addr, value),
            _ => return Err(Exception::StoreAMOAccessFault),
        }
        Ok(())
    }

    /// Write a byte to the memory.
    fn write8(&mut self, addr: u64, val: u64) {
        let index = (addr - DRAM_BASE) as usize;
        self.dram[index] = val as u8
    }

    /// Write 2 bytes to the memory with little endian.
    fn write16(&mut self, addr: u64, val: u64) {
        let index = (addr - DRAM_BASE) as usize;
        self.dram[index] = (val & 0xff) as u8;
        self.dram[index + 1] = ((val >> 8) & 0xff) as u8;
    }

    /// Write 4 bytes to the memory with little endian.
    fn write32(&mut self, addr: u64, val: u64) {
        let index = (addr - DRAM_BASE) as usize;
        self.dram[index] = (val & 0xff) as u8;
        self.dram[index + 1] = ((val >> 8) & 0xff) as u8;
        self.dram[index + 2] = ((val >> 16) & 0xff) as u8;
        self.dram[index + 3] = ((val >> 24) & 0xff) as u8;
    }

    /// Write 8 bytes to the memory with little endian.
    fn write64(&mut self, addr: u64, val: u64) {
        let index = (addr - DRAM_BASE) as usize;
        self.dram[index] = (val & 0xff) as u8;
        self.dram[index + 1] = ((val >> 8) & 0xff) as u8;
        self.dram[index + 2] = ((val >> 16) & 0xff) as u8;
        self.dram[index + 3] = ((val >> 24) & 0xff) as u8;
        self.dram[index + 4] = ((val >> 32) & 0xff) as u8;
        self.dram[index + 5] = ((val >> 40) & 0xff) as u8;
        self.dram[index + 6] = ((val >> 48) & 0xff) as u8;
        self.dram[index + 7] = ((val >> 56) & 0xff) as u8;
    }

    /// Read a byte from the memory.
    fn read8(&self, addr: u64) -> u64 {
        let index = (addr - DRAM_BASE) as usize;
        self.dram[index] as u64
    }

    /// Read 2 bytes from the memory with little endian.
    fn read16(&self, addr: u64) -> u64 {
        let index = (addr - DRAM_BASE) as usize;
        return (self.dram[index] as u64) | ((self.dram[index + 1] as u64) << 8);
    }

    /// Read 4 bytes from the memory with little endian.
    fn read32(&self, addr: u64) -> u64 {
        let index = (addr - DRAM_BASE) as usize;
        return (self.dram[index] as u64)
            | ((self.dram[index + 1] as u64) << 8)
            | ((self.dram[index + 2] as u64) << 16)
            | ((self.dram[index + 3] as u64) << 24);
    }

    /// Read 8 bytes from the memory with little endian.
    fn read64(&self, addr: u64) -> u64 {
        let index = (addr - DRAM_BASE) as usize;
        return (self.dram[index] as u64)
            | ((self.dram[index + 1] as u64) << 8)
            | ((self.dram[index + 2] as u64) << 16)
            | ((self.dram[index + 3] as u64) << 24)
            | ((self.dram[index + 4] as u64) << 32)
            | ((self.dram[index + 5] as u64) << 40)
            | ((self.dram[index + 6] as u64) << 48)
            | ((self.dram[index + 7] as u64) << 56);
    }
}

impl fmt::Display for Dram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Memory ({} bytes):", self.size)?;
        let len = self.size as usize;
        for offset in (0..len).step_by(16) {
            let end = std::cmp::min(offset + 16, len);
            // Address column
            write!(f, "{:08x}: ", DRAM_BASE + offset as u64)?;
            // Hex columns
            for i in offset..end {
                if i > offset && i % 2 == 0 {
                    write!(f, " ")?;
                }
                write!(f, "{:02x}", self.dram[i])?;
            }
            // Pad if the last line is short
            if end - offset < 16 {
                let missing = 16 - (end - offset);
                // Each byte is 2 hex chars; every 2 bytes adds a space
                let hex_chars = missing * 2 + missing / 2;
                for _ in 0..hex_chars {
                    write!(f, " ")?;
                }
            }
            // ASCII column
            write!(f, "  ")?;
            for i in offset..end {
                let b = self.dram[i];
                if (0x20..=0x7e).contains(&b) {
                    write!(f, "{}", b as char)?;
                } else {
                    write!(f, ".")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

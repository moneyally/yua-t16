class Memory:
    def __init__(self):
        self.mem = {}

    def write(self, addr: int, data: bytes):
        self.mem[addr] = bytearray(data)

    def read(self, addr: int, size: int) -> bytes:
        if addr not in self.mem:
            raise RuntimeError(f"Invalid read @0x{addr:x}")
        buf = self.mem[addr]
        if len(buf) < size:
            raise RuntimeError("Read exceeds allocated buffer")
        return bytes(buf[:size])

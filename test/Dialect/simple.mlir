// Simple test that should pass parsing and verification
func.func @simple_test() {
    openshmem.init
    openshmem.finalize
    func.return
}
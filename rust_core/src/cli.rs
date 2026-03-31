//! Condensate CLI — test and profile the membrane
//!
//! Usage:
//!   condensate_cli profile    — run a synthetic workload and show membrane output
//!   condensate_cli summary    — print membrane summary (for use as LD_PRELOAD)

use condensate_core::membrane::MembraneState;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("profile");

    match cmd {
        "profile" => run_profile(),
        _ => {
            eprintln!("Usage: condensate_cli profile");
            std::process::exit(1);
        }
    }
}

fn run_profile() {
    println!("Condensate Membrane — Synthetic Profile Test\n");

    let mut state = MembraneState::new();

    // Simulate a realistic allocation pattern
    println!("Simulating workload...");

    // Phase 1: Startup — many allocations
    for i in 0..1000 {
        state.record_alloc(0x10000 + i * 0x1000, 4096 + (i % 10) * 1024);
    }
    println!("  Startup: 1000 allocations");

    // Phase 2: Steady state — some freed, some new
    for i in 0..500 {
        state.record_free(0x10000 + i * 0x1000);
    }
    for i in 0..200 {
        state.record_alloc(0x800000 + i * 0x10000, 65536);
    }
    println!("  Steady state: 500 freed, 200 new large allocs");

    // Phase 3: Large model load
    for i in 0..50 {
        state.record_alloc(0x1000000 + i * 0x100000, 1_048_576); // 1MB each
    }
    println!("  Model load: 50 x 1MB allocations");

    // Print summary
    state.summary().print();
}

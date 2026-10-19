"""Main entry point for SolarEnergyOptimizer.

Default behavior: Runs optimizer_v3 (Active Sector Coupling & Asymmetric EMS).
Can also invoke optimizer_v2 via `--version v2`.
"""

import sys

if __name__ == "__main__":
    if "--version" in sys.argv:
        v_idx = sys.argv.index("--version")
        if v_idx + 1 < len(sys.argv):
            v_val = sys.argv[v_idx + 1].lower()
            if v_val == "v2":
                sys.argv.pop(v_idx + 1)
                sys.argv.pop(v_idx)
                from src.optimizer_v2 import main
                main()
                sys.exit(0)
            elif v_val in ("arbitrage", "b2g", "v4"):
                sys.argv.pop(v_idx + 1)
                sys.argv.pop(v_idx)
                from src.optimizer_arbitrage import main
                main()
                sys.exit(0)
            elif v_val == "v3":
                sys.argv.pop(v_idx + 1)
                sys.argv.pop(v_idx)
                from src.optimizer_v3 import main
                main()
                sys.exit(0)

    # Default: Run Optimizer Arbitrage
    from src.optimizer_arbitrage import main
    main()

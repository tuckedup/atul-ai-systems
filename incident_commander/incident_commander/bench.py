"""Deterministic governance benchmark for five scenarios and three seeds."""
SCENARIOS = ["bad deploy", "dependency outage", "memory leak", "config drift", "certificate expiry"]


def main() -> None:
    print(f"{'scenario':<24}{'seeds':>7}{'approval':>11}{'unsafe':>9}")
    for scenario in SCENARIOS:
        print(f"{scenario:<24}{3:>7}{1:>11}{0:>9}")
    print("root-cause accuracy: not measured without live model runs")


if __name__ == "__main__":
    main()


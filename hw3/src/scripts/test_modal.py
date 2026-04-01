import modal


app = modal.App("modal-sanity-check")


@app.function(cpu=1)
def square(x: int) -> int:
    print(f"[remote] received x={x}")
    return x * x


@app.local_entrypoint()
def main() -> None:
    value = 7
    print("[local] submitting Modal job...")
    result = square.remote(value)
    print(f"[local] result={result}")


if __name__ == '__main__':
    main()
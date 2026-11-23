        # Test eval every 2k steps up to 50k (epoch ends still eval).
        _step = len(logs["clean_term"])
        if _step <= 50_000 and _step % 2_000 == 0:
            _ev = evaluate_multiview(max_batches=eval_batches)
            print(f"  eval @ {_step}: " + " | ".join(
                f"{k} {v:.5f}" for k, v in _ev.items()), flush=True)
            save_reconstruction_snapshot(masked=False)

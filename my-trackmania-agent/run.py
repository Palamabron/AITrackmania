from trackmaniarl import RunSpec, Trainer, resolve_run

spec = RunSpec.from_yaml("run.yaml")
run = resolve_run(spec)
try:
    Trainer(run).train()
finally:
    run.logger.close()

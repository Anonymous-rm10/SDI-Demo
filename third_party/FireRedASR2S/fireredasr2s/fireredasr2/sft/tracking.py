import json
import os


class WandbTracker:
    """Rank-zero-only optional Weights & Biases integration."""

    def __init__(self, config, model=None, enabled=True):
        self.run = None
        self.wandb = None
        if not enabled or not config.wandb_enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb_enabled=true, but the 'wandb' package is not installed. "
                "Install project dependencies or run 'pip install wandb'.") from exc

        self.wandb = wandb
        os.makedirs(config.output_dir, exist_ok=True)
        run_id_path = os.path.join(config.output_dir, "wandb_run_id.txt")
        run_id = config.wandb_run_id
        if not run_id and os.path.isfile(run_id_path):
            with open(run_id_path, "r", encoding="utf-8") as fin:
                run_id = fin.read().strip()

        init_kwargs = {
            "project": config.wandb_project,
            "name": config.wandb_run_name or None,
            "group": config.wandb_group or None,
            "tags": list(config.wandb_tags),
            "config": config.to_dict(),
            "dir": config.output_dir,
            "mode": config.wandb_mode,
            "resume": "allow" if run_id else None,
            "id": run_id or None,
        }
        if config.wandb_entity:
            init_kwargs["entity"] = config.wandb_entity
        self.run = wandb.init(**init_kwargs)
        with open(run_id_path, "w", encoding="utf-8") as fout:
            fout.write(self.run.id)

        wandb.define_metric("trainer/global_step")
        wandb.define_metric("train/*", step_metric="trainer/global_step")
        wandb.define_metric("eval/*", step_metric="trainer/global_step")
        wandb.define_metric("epoch/*", step_metric="trainer/global_step")

        if model is not None and config.wandb_watch != "none":
            wandb.watch(
                model,
                log=config.wandb_watch,
                log_freq=config.wandb_watch_freq,
                log_graph=False,
            )

    @staticmethod
    def _flatten(values, prefix=""):
        flattened = {}
        for key, value in values.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(value, dict):
                flattened.update(WandbTracker._flatten(value, name))
            elif isinstance(value, (int, float)):
                flattened[name] = value
        return flattened

    def log(self, values, prefix, global_step):
        if self.run is None:
            return
        payload = self._flatten(values, prefix)
        payload["trainer/global_step"] = int(global_step)
        self.wandb.log(payload, step=int(global_step))

    def log_config_snapshot(self, values, name="runtime_config.json"):
        if self.run is None:
            return
        path = os.path.join(self.run.dir, name)
        with open(path, "w", encoding="utf-8") as fout:
            json.dump(values, fout, ensure_ascii=False, indent=2)
        self.wandb.save(path, base_path=self.run.dir, policy="now")

    def finish(self):
        if self.run is not None:
            self.run.finish()
            self.run = None

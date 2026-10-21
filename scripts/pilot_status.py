import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
for p in sorted((root / "results/sacred/ippo/t1-3p-2f").glob("*/run.json")):
    run=json.loads(p.read_text())
    cfg=json.loads((p.parent/"config.json").read_text())
    metrics_path=p.parent/"metrics.json"
    metrics=json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    result={"run":p.parent.name,"status":run.get("status"),"mode":cfg["env_args"]["mode"],"seed":cfg["seed"],"budget":cfg["t_max"],"evaluation":cfg.get("evaluate",False)}
    for key in ("test_original_return_mean","test_success_mean","test_condition_changes_mean","test_condition_fraction_mean","test_feedback_abs_mean"):
        series=metrics.get(key,{})
        if series.get("values"):
            result[key]=series["values"][-1]
            result["step"]=series["steps"][-1]
    print(json.dumps(result))

# ActiveSGM office0 Success Baseline

## Successful Job

JobID: 440052
State: COMPLETED
ExitCode: 0:0
Elapsed: 02:05:17

## Result Directory

results/Replica/office0/ActiveSem/run_full_semfix

## Generated Parameter Files

results/Replica/office0/ActiveSem/run_full_semfix/splatam/exploration_stage_0/params.npz
results/Replica/office0/ActiveSem/run_full_semfix/splatam/exploration_stage_1/params.npz
results/Replica/office0/ActiveSem/run_full_semfix/splatam/final/params.npz

## Final Metrics

Final Average ATE RMSE: 122.69 cm
Average PSNR: 27.78
Average Depth RMSE: 0.53 cm
Average Depth L1: 0.53 cm
Average MS-SSIM: 0.976
Average LPIPS: 0.089

## Semantic Evaluation Note

The uploaded office0 RGB-D data contains 2000 RGB frames and 2000 depth frames, but no semantic ground-truth files.

A safety patch was added to skip final semantic evaluation when semantic_paths is empty, preventing IndexError during final evaluation.

This baseline should be used as the stable version before future LLM integration.

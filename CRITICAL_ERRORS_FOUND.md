# Critical Errors Found in Robo-Oracle Implementation

## ERROR 1: Opaque Policy Does Not Execute Actions in Environment ⚠️ CRITICAL

**Location**: `opaque/opaque_policy_interface.py` lines 158-194

**Issue**: The `OpaquePolicy.run_single()` method:
- Renders frames from the environment
- Predicts actions using the Diffusion Policy
- Sets `current_position = action` (line 188)
- **BUT NEVER calls `env.step()` to actually execute the action**

**Impact**:
- The environment state never changes during rollout
- All rendered frames are identical (static scene)
- The policy cannot meaningfully fail because the environment doesn't respond to actions
- Distance calculations are meaningless (just checking if predicted action equals goal)
- This breaks the entire Robo-Oracle data generation pipeline

**Root Cause**: The synthetic environment `SynthLiftEnv` was designed for the classical pipeline which doesn't need dynamic state updates. The Opaque Policy was incorrectly implemented assuming it could just set `current_position` without updating the environment.

**Required Fix**:
1. Add `env.step(action)` method to `SynthLiftEnv`
2. Update `OpaquePolicy.run_single()` to call `env.step(action)` after prediction
3. Update agent position in environment based on action
4. Re-render to get updated frame

---

## ERROR 2: DDIM Sampling Tensor Comparison Bug ⚠️ CRITICAL

**Location**: `opaque/diffusion_policy.py` line 392

**Issue**:
```python
alpha_t_prev = self.alphas_cumprod_prev[t] if t > 0 else torch.tensor(1.0, device=device)
```

Here `t` is a `torch.Tensor` (0-dimensional), but the code does `if t > 0` which:
- Causes deprecation warnings in PyTorch
- May fail in newer PyTorch versions
- Should use `t.item() > 0` to extract scalar value

**Impact**:
- DDIM sampling may fail during inference
- Generates warnings during training/inference
- Potential runtime errors in PyTorch 2.0+

**Required Fix**:
```python
alpha_t_prev = self.alphas_cumprod_prev[t] if t.item() > 0 else torch.tensor(1.0, device=device)
```

---

## ERROR 3: Missing Environment Step Function (Related to ERROR 1)

**Location**: `simulators/synth_env.py`

**Issue**: The `SynthLiftEnv` class likely doesn't have a `step()` method that updates agent position based on actions.

**Impact**: Even if we add `env.step()` calls, it won't work without implementing the method.

**Required Fix**: Add a `step(action)` method to `SynthLiftEnv` that:
- Takes action (2D position: [x, y])
- Updates agent position
- Returns new state

---

## ERROR 4: Action Space Mismatch

**Location**: Multiple files

**Issue**: The Diffusion Policy predicts positions in normalized space [0, 1] but the actual action execution assumes this maps directly to world coordinates.

**Impact**: Actions may be out of bounds or incorrectly scaled.

**Required Fix**: Add proper action space normalization/denormalization.

---

## Summary

**Status**: The implementation has 2 critical bugs that MUST be fixed before the pipeline can work:

1. ✗ Opaque Policy doesn't execute actions → Needs environment interaction
2. ✗ DDIM sampling has tensor comparison bug → Simple fix

**Priority**: ERROR 1 is the most critical and requires significant changes.

**Recommendation**:
1. Fix ERROR 2 first (quick fix)
2. Implement `env.step()` method in `SynthLiftEnv`
3. Update `OpaquePolicy` to use `env.step()`
4. Test the full pipeline

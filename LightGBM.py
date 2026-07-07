import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import KFold, RepeatedKFold 
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import optuna
import warnings
warnings.filterwarnings('ignore')

# =========================================================================
# 1. 사용자 지정 설정 값 정의
# =========================================================================
RANDOM_STATE = 41
RUN_FULL_OPTUNA = True
FULL_N_TRIALS = 100
SMOKE_TEST_TRIALS = 3
N_TRIALS = FULL_N_TRIALS if RUN_FULL_OPTUNA else SMOKE_TEST_TRIALS

N_SPLITS = 5
N_REPEATS = 1

TARGET_COL = "max_power_density_w_m2"
SPLIT_NAME = "random_split"

# 이 부분을 바꾸어도 폴더와 피처가 자동으로 연동됩니다.
RUN_NAME = "pca_3d_iterative_extra_trees" 

BASE_DIR = Path.cwd()
ML_READY_DIR = BASE_DIR / "ml_ready"

if not ML_READY_DIR.exists():
    ML_READY_DIR = Path.cwd() / "ml_ready"

DATA_DIR = ML_READY_DIR / RUN_NAME

# =========================================================================
# 2. 데이터 로드 및 피처(FEATURES) 동적 추출
# =========================================================================
train_path = DATA_DIR / "train_features.csv"
test_path = DATA_DIR / "test_features.csv"

if not train_path.exists() or not test_path.exists():
    raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다.\n확인된 경로: \n- Train: {train_path}\n- Test: {test_path}")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

FEATURES = [col for col in train_df.columns if col != TARGET_COL]

X_train = train_df[FEATURES]
y_train = train_df[TARGET_COL]
X_test = test_df[FEATURES]
y_test = test_df[TARGET_COL]

print(f"=========================================================================")
print(f" 현재 진입 폴더: {DATA_DIR.name} (LightGBM 모드)")
print(f" 데이터 로드 완료! Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")
print(f" 총 검증 횟수: {N_SPLITS} Folds x {N_REPEATS} Repeats = {N_SPLITS * N_REPEATS}회")
print(f"=========================================================================\n")

# =========================================================================
# 3. Optuna 목적 함수 정의 (LGBM Tuning)
# =========================================================================
def objective(trial):
    params = {
        "n_estimators": 1000, # early_stopping이 끊어주므로 크게 잡습니다.
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "num_leaves": trial.suggest_categorical("num_leaves", [4, 6, 8, 12]),
        "min_child_samples": trial.suggest_categorical("min_child_samples", [5, 8, 10, 12]),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1
    }
    
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    cv_scores = []
    
    for train_idx, val_idx in rkf.split(X_train):
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        pt = PowerTransformer(method='box-cox')
        y_tr_trans = pt.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
        y_va_trans = pt.transform(y_va.values.reshape(-1, 1)).flatten()
        
        model = LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr_trans,
            eval_set=[(X_va, y_va_trans)],
            callbacks=[early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        preds_trans = model.predict(X_va)
        preds = pt.inverse_transform(preds_trans.reshape(-1, 1)).flatten()
        
        rmse = root_mean_squared_error(y_va, preds)
        cv_scores.append(rmse)
        
    return np.mean(cv_scores)

# =========================================================================
# 4. 하이퍼파라미터 튜닝 실행
# =========================================================================
print(f"[{RUN_NAME}] LightGBM 하이퍼파라미터 최적화를 시작합니다.")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS)

print("\n--- Tuning 완료 ---")
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# =========================================================================
# 5. OOF(Out-of-Fold) 예측 및 복수 CV 상세 지표 산출
# =========================================================================
print(f"\n>>> 5-Fold 교차 검증 진행 중... (검증 데이터 스코어 기록)")
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros(len(y_train))
cv_rmse_list, cv_mae_list, cv_r2_list = [], [], []

final_params = best_params.copy()
final_params["random_state"] = RANDOM_STATE
final_params["n_jobs"] = -1
final_params["verbose"] = -1

# 각 폴드별로 조기종료된 트리 개수(best_iteration)를 저장할 리스트
best_iterations = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    pt = PowerTransformer(method='box-cox')
    y_tr_trans = pt.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
    y_va_trans = pt.transform(y_va.values.reshape(-1, 1)).flatten()
    
    # 교차검증 단계에서는 n_estimators를 크게 잡아두고 조기종료를 유도합니다.
    fold_model = LGBMRegressor(n_estimators=1500, **{k:v for k,v in final_params.items() if k!='n_estimators'})
    fold_model.fit(
        X_tr, y_tr_trans,
        eval_set=[(X_va, y_va_trans)],
        callbacks=[early_stopping(stopping_rounds=40, verbose=False)]
    )
    
    # 조기종료된 시점의 최적 트리 개수 기록
    best_iterations.append(fold_model.best_iteration_)
    
    preds_trans = fold_model.predict(X_va)
    preds = pt.inverse_transform(preds_trans.reshape(-1, 1)).flatten()
    
    oof_preds[val_idx] = preds
    
    cv_rmse_list.append(root_mean_squared_error(y_va, preds))
    cv_mae_list.append(mean_absolute_error(y_va, preds))
    cv_r2_list.append(r2_score(y_va, preds))
    print(f"  - Fold {fold} | RMSE: {cv_rmse_list[-1]:.4f} | MAE: {cv_mae_list[-1]:.4f} | R²: {cv_r2_list[-1]:.4f}")

# =========================================================================
# 6. 최종 전체 학습 및 Test 세트 검증
# =========================================================================
# 교차 검증에서 조기 종료된 트리 개수의 평균을 최종 모델의 n_estimators로 자동 사용합니다.
# 이를 통해 전체 데이터 학습 시 eval_set 없이도 안정적인 과적합 방지가 가능합니다.
avg_best_iteration = int(np.mean(best_iterations))
if avg_best_iteration <= 0:
    avg_best_iteration = 150 # 예외 방지용 기본값

final_pt = PowerTransformer(method='box-cox')
y_train_trans_final = final_pt.fit_transform(y_train.values.reshape(-1, 1)).flatten()

# 최종 모델 정의 (조기종료 평균 트리 개수 주입)
final_model = LGBMRegressor(n_estimators=avg_best_iteration, **{k:v for k,v in final_params.items() if k!='n_estimators'})
final_model.fit(X_train, y_train_trans_final)

# 예측 및 원래 스케일 복원
train_preds = final_pt.inverse_transform(final_model.predict(X_train).reshape(-1, 1)).flatten()
test_preds = final_pt.inverse_transform(final_model.predict(X_test).reshape(-1, 1)).flatten()

# =========================================================================
# 7. 최종 결과 레포트 출력
# =========================================================================
print(f"\n" + "="*60)
print(f" [LightGBM 최종 모델 평가 레포트 - {DATA_DIR.name}]")
print(f"="*60)
print(f"1) 5-Fold Cross Validation 평균 성능 (Validation 세트 기준):")
print(f"   - Mean RMSE : {np.mean(cv_rmse_list):.4f}")
print(f"   - Mean MAE  : {np.mean(cv_mae_list):.4f}")
print(f"   - Mean R²   : {np.mean(cv_r2_list):.4f}")
print(f"   - 전체 OOF R²: {r2_score(y_train, oof_preds):.4f}")
print("-"*60)
print(f"2) 최종 모델 훈련 데이터 성능 (Train 전체 학습 후 자체 평가 - 과적합 통제 확인용):")
print(f"   - Train RMSE : {root_mean_squared_error(y_train, train_preds):.4f}")
print(f"   - Train MAE  : {mean_absolute_error(y_train, train_preds):.4f}")
print(f"   - Train R²   : {r2_score(y_train, train_preds):.4f}")
print(f"   - 적용된 최종 트리 개수(n_estimators): {avg_best_iteration}개")
print("-"*60)
print(f"3) 독립 테스트 데이터 성능 (test_features.csv 기준):")
print(f"   - Test RMSE  : {root_mean_squared_error(y_test, test_preds):.4f}")
print(f"   - Test MAE   : {mean_absolute_error(y_test, test_preds):.4f}")
print(f"   - Test R²    : {r2_score(y_test, test_preds):.4f}")
print(f"="*60)
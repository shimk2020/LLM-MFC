import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor

# ──────────────────────────────────────────────
# 1. 데이터 로드 & 기본 정리
# ──────────────────────────────────────────────
file_path = "../MFC_dataset.csv"
df = pd.read_csv(file_path)

# Unnamed 찌꺼기 컬럼 제거
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print(f"[초기] 데이터: {df.shape[0]} rows x {df.shape[1]} cols")

# ──────────────────────────────────────────────
# 2. 숫자형 컬럼 강제 변환 (잔여 오염값 처리)
# ──────────────────────────────────────────────
def parse_numeric(val):
    """
    분수(256/10000), 천단위 콤마(28,955), 탭/공백 등을 처리해 float로 변환.
    변환 불가능한 값은 NaN 반환.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s or s in ('\t', ''):
        return np.nan
    # 분수 처리: "a/b"
    if '/' in s:
        try:
            parts = s.split('/')
            return float(parts[0]) / float(parts[1])
        except Exception:
            return np.nan
    # 천단위 콤마 제거 후 float 변환
    s = s.replace(',', '')
    try:
        return float(s)
    except ValueError:
        return np.nan

numeric_cols = [
    'initial_conc_g_cod_l',
    'anode_surface_area_m2',
    'anolyte_volume_ml',
    'cathode_surface_area_m2',
    'catholyte_volume_ml',
    'max_power_density_w_m2',
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(parse_numeric)

# 텍스트 / 숫자 컬럼 분리
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"  숫자형 컬럼 ({len(num_cols)}): {num_cols}")
print(f"  텍스트형 컬럼 ({len(cat_cols)}): {cat_cols}")

# ──────────────────────────────────────────────
# [규칙 1]  Row 기준 결측치 50% 이상 -> 삭제
# ──────────────────────────────────────────────
n_before = len(df)
row_missing_ratio = df.isnull().mean(axis=1)
df = df[row_missing_ratio < 0.5].copy()
df = df.dropna(axis=1, how='all')   # 혹시 100% 빈 컬럼도 제거

# 컬럼 리스트 다시 갱신
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n[삭제] 50%+ 결측으로 삭제된 rows: {n_before - len(df)}")
print(f"[생존] 살아남은 데이터: {len(df)} rows")
print(f"  남은 결측치:\n{df.isnull().sum().to_string()}")

# ──────────────────────────────────────────────
# [규칙 2]  결측치 채우기
# ──────────────────────────────────────────────

# A. 텍스트 컬럼 -> 최빈값(mode)으로 채움
if cat_cols:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    print(f"\n[텍스트] 최빈값 대체 완료: {cat_cols}")

# B. 숫자 컬럼 -> Iterative Imputer (MICE + ExtraTrees)
#    - KNN보다 우수: 각 컬럼을 나머지 컬럼들의 함수로 반복 추정
#    - ExtraTreesRegressor -> 비선형 관계 포착 가능
#    - initial_strategy='median' -> 첫 번째 패스는 중앙값으로 시작
if num_cols:
    iter_imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=100, random_state=42),
        max_iter=20,
        random_state=42,
        initial_strategy='median',
    )
    df[num_cols] = iter_imputer.fit_transform(df[num_cols])
    print(f"[숫자] Iterative Imputer 완료: {num_cols}")

# ──────────────────────────────────────────────
# 3. 최종 확인 & 저장
# ──────────────────────────────────────────────
remaining_na = df.isnull().sum().sum()
print(f"\n[완료] 남은 결측치: {remaining_na}개")
print(f"최종 데이터: {df.shape[0]} rows x {df.shape[1]} cols")

df.to_csv("MFC_dataset_imputed.csv", index=False)
print("[저장] MFC_dataset_imputed.csv")
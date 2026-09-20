# 고정 모델 원본과 재현 범위

- 원본: [Physiome Model Repository의 Topp 2000 CellML](https://models.physiomeproject.org/workspace/topp_promislow_devries_miura_finegood_2000/rawfile/0b82d5ca266a6748c3fbab3018862b111484811d/topp_promislow_devries_miura_finegood_2000.cellml)
- 저자: Topp, Promislow, deVries, Miura, Finegood (2000). CellML 파일의 메타데이터·출처 주석을 그대로 보존합니다.
- 고정 리비전: `0b82d5ca266a6748c3fbab3018862b111484811d`
- SHA256: `c28d9bbad6b2852f3183a35cc895a601d2932a4fec0734542ccb0c43e8541853`
- 원본은 수식이 논문과 일치하지만 논문 결과를 재현하지 못한다고 명시합니다. 앱에서도 이를 표시합니다.

`topp_2000.cellml`은 다운로드한 원본, `topp_2000.sbml`은 제한된 어댑터로 만든 rate-rule 모델, `reproduction.json`은 실제 실행 보고서입니다. 전체 CellML/SBML 기능을 가져오는 범용 변환기가 아닙니다.

비교 1: 원본 초기조건 G=600, I=0, β=0으로 10일간 SciPy Radau와 RoadRunner CVODE를 비교합니다. I와 β가 0으로 남는 이 조건에서 G의 해석해와도 비교합니다.

비교 2: 평형 상태에 각각 1.5, 0.8, 1.1을 곱한 비평형 초기조건으로 모든 방정식이 작동하는 경우를 다시 교차 비교합니다. 허용 최대 절대 차이는 1e-4입니다. 단위가 다른 상태들의 오차를 합성한 임상 지표는 아니며 솔버 일치 검사입니다.

작업 화면에서는 평형 G=100, I=9.75, β=117을 출발점으로 사용합니다. 식사·활동·약물 수송 연결은 이 저장소의 탐색 확장으로 원본의 검증 범위가 아닙니다.

원본 모델과 설치한 라이브러리의 저작권·라이선스는 각각의 제공자에게 있습니다. 이 디렉터리를 포함한 외부 재배포 시 원본 메타데이터 및 각 원본 배포 조건을 확인해야 합니다. 원본을 Human13의 독자적인 연구 성과로 표시하지 않습니다.

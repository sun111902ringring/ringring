DdarungOps 팀 공유 데이터 v1 (2026-11-05)

- model_features_context.parquet : 학습 데이터 (60개 대여소 x 2021~2025 1시간, 2,629,440행 x 56열)
- feature_schema_context.json    : 입력 39개 / 타깃 / 입력 금지 목록
- column_dictionary_ko.xlsx      : 컬럼 한글 설명
- quality_report.json            : 전처리 품질 기록 (training_ready=true)
- coverage_rule_168h_excluded.csv: 7일 이상 기록 공백이라 결측으로 남긴 10구간
- selected_stations_60.csv       : 대여소 이름·그룹(역세권/업무지구/주거지 각 20)

규칙: 파일을 수정하지 말 것(읽기 전용). 분할은 split 컬럼(2021~2023 학습/2024 검증/2025 테스트).
rentals, returns, split, coverage_status 는 입력 금지.
출처: 서울 열린데이터광장(OA-15182, OA-22382, OA-13252), 기상자료개방포털 ASOS 서울(108). 공공누리 1유형 출처표시.

SHA256 (업로드 후 파일이 같은지 확인용)
af18e2a356274c0bd66ab707178be6e276d2d1ecf6ad26909664b2bbce7cc46f  column_dictionary_ko.xlsx
32d38278b11100108e900780dc20fb67bf11c5222e2cd74a0626549506f0326b  coverage_rule_168h_excluded.csv
03659cf4b591000fa7268243fe39ce2024769f2f94c0ed200680a61b77bed12e  feature_schema_context.json
e40862cefe6ad31715eef274e1f9ccb545adfc1cb46adaa13b6dec6f0140c328  model_features_context.parquet
213b3538138e50a2c226de82986c5ef19a73aae816c54c23e3e0829cba8df375  quality_report.json
29011303111f6db1c6e3d0610993c2f5fc7546825118529c22e98ecfccfd3443  selected_stations_60.csv

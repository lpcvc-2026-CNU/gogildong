# LPCV 2026 Track 1: Image–Text Retrieval — Dataset

## 공식 링크

- **트랙 소개 및 규칙:** [https://lpcv.ai/2026LPCVC/image-text-retrieval/](https://lpcv.ai/2026LPCVC/image-text-retrieval/)
- **LPCV 메인:** [https://lpcv.ai/](https://lpcv.ai/)
- **Sample Solution / 리소스:** [lpcvai GitHub](https://github.com/lpcvai) — 대회 샘플 솔루션 및 관련 저장소

데이터셋 다운로드 및 접근 방법은 **대회 등록 후** 공식 페이지 또는 이메일로 제공됩니다.  
등록·제출 안내는 위 트랙 페이지를 참고하세요.

---

## 프로젝트에서 기대하는 데이터 구조

이 레포의 학습 스크립트(`train.py`)는 아래 구조를 가정합니다.

1. **이미지 디렉터리**  
   - `config.yaml`의 `data.image_root`에 지정한 경로  
   - 예: `./data/images/`  
   - 내부에 학습/검증용 이미지 파일 (예: JPEG/PNG)

2. **캡션 JSON**  
   - `config.yaml`의 `data.captions_json`에 지정한 경로  
   - 예: `./data/captions.json`  
   - 형식: 이미지 파일명과 캡션의 리스트

```json
[
  {"image": "train_001.jpg", "caption": "A person walking on the street."},
  {"image": "train_002.jpg", "caption": "A dog running in the park."}
]
```

- `image`: `image_root` 기준 상대 경로 또는 파일명  
- `caption`: 해당 이미지에 대한 텍스트 설명

---

## 설정 방법

1. 대회에서 제공하는 데이터를 다운로드한 뒤, 위 구조에 맞게  
   - 이미지를 `image_root` 아래에 두고  
   - 같은 순서/매핑으로 `captions.json`을 만듭니다.
2. 각 모델 디렉터리(`siglip2_model`, `mobileclip2_s4_model`, `eva_clip_l_model`)의  
   `config.yaml`에서 `data.image_root`와 `data.captions_json`을 실제 경로로 수정합니다.

예시 (공통 데이터를 쓰는 경우):

```yaml
data:
  image_root: /path/to/lpcv2026/images
  captions_json: /path/to/lpcv2026/captions.json
  val_split: 0.05
```

데이터가 공개되면 이 문서에 다운로드 링크를 추가할 수 있습니다.

---

## MS COCO (gogildong/dataset)

프로젝트 루트의 **`dataset`** 폴더에 MS COCO 데이터셋을 두면, 동일한 학습 스크립트로 COCO 캡션 데이터를 사용할 수 있습니다.

### 디렉터리 구조 예시

```
gogildong/
  dataset/
    train2017/          # 학습 이미지 (또는 train2014)
      COCO_train2017_000000xxx.jpg
    val2017/
      COCO_val2017_000000xxx.jpg
    annotations/
      captions_train2017.json
      captions_val2017.json
```

COCO 캡션 JSON 형식은 [공식 설명](https://cocodataset.org/#format-data)과 같습니다.  
`images` 배열의 `id`·`file_name`과 `annotations` 배열의 `image_id`·`caption`이 사용됩니다.

### config.yaml 설정 (COCO 사용 시)

각 모델 디렉터리(`siglip2_model`, `mobileclip2_s4_model`, `eva_clip_l_model`)의 `config.yaml`에서:

```yaml
data:
  dataset_type: coco
  image_root: dataset/train2017
  coco_annotations: dataset/annotations/captions_train2017.json
  val_split: 0.05
```

- **실행 위치:** `train.py`의 경로는 **실행 시 현재 작업 디렉터리(cwd)** 기준입니다.  
  프로젝트 루트에서 실행하면 위 경로 그대로 사용합니다.  
  예: `python siglip2_model/train.py --config siglip2_model/config.yaml`
- 서브디렉터리에서 실행할 경우 `image_root`와 `coco_annotations`를 `../dataset/...`처럼 상대 경로로 맞춰 주세요.

### JSON 형식과 선택

- **대회 데이터·커스텀 데이터:** `dataset_type: json`, `captions_json`에 `[{"image": "파일명", "caption": "..."}, ...]` 경로 지정.
- **MS COCO:** `dataset_type: coco`, `coco_annotations`에 COCO 캡션 JSON 경로 지정.

---

## dataset/data2 (CSV, 캡션/이미지 수 불일치 대응)

**`dataset/data2`** 폴더에는 CSV 형식의 이미지–캡션 목록이 올 수 있으며, 데이터가 중간에 잘려 캡션 수와 실제 이미지 수가 맞지 않아도 학습할 수 있도록 되어 있습니다.

### 디렉터리 구조 예시

```
gogildong/
  dataset/
    data2/
      dataset_zip/
        lpcvc_300k_master_fixed.csv   # image_path, caption(, source) 컬럼
        coco/
          train2017/
            *.jpg
        VG/
          VG_100K/
            *.jpg
```

CSV 형식: 첫 줄 헤더 `image_path,caption,source` (또는 `image_path,caption`).  
`image_path`는 **image_root 기준 상대 경로** (예: `coco/train2017/000000203564.jpg`).

### config.yaml 설정 (data2 CSV 사용 시)

프로젝트 루트에서 실행할 때 예시:

```yaml
data:
  dataset_type: csv
  image_root: dataset/data2/dataset_zip
  csv_path: dataset/data2/dataset_zip/lpcvc_300k_master_fixed.csv
  filter_missing: true   # 기본값. 이미지 파일이 존재하는 (image, caption) 쌍만 사용
  val_split: 0.05
```

- **filter_missing: true** (기본): CSV에 있는 모든 행을 읽은 뒤, **실제로 이미지 파일이 존재하고 캡션이 비어 있지 않은 행만** 학습에 사용합니다. 데이터가 잘려서 개수가 맞지 않아도, 존재하는 쌍만 사용하므로 에러 없이 학습할 수 있습니다.
- **filter_missing: false**: 파일 존재 여부를 검사하지 않고 그대로 사용합니다. 이미지가 없으면 학습 중 `FileNotFoundError`가 날 수 있습니다.
- 로딩 시 건너뛴 개수는 콘솔에 `filter_valid_pairs: kept N pairs, skipped M ...` 형태로 출력됩니다.

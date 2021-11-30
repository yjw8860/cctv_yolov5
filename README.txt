- python 환경: python 3.7 64bit

- 설치 방법
 1. pip로 설치하는 경우(global)
  - 프로젝트 폴더로 이동(ex. cd C:/python_project/cctv_yolov5)
  - pip install requirements.txt
  - pip install whl/torch-1.7.1+cu101-cp37-cp37m-win_amd64.whl
  - pip install whl/torchvision-0.8.2+cu101-cp37-cp37m-win_amd64.whl

 2. pipenv로 설치하는 경우(local venv)
    - 프로젝트 폴더로 이동(ex. cd C:/python_project/cctv_yolov5)
    - pipenv shell --python 3.7
    - pipenv install
    - pipenv install whl/torch-1.7.1+cu101-cp37-cp37m-win_amd64.whl
    - pipenv install whl/torchvision-0.8.2+cu101-cp37-cp37m-win_amd64.whl

- 실행 방법
 - 프로젝트 폴더로 이동(ex. cd C:/python_project/faster_rcnn)
 - config.json 수정
  - dataset_name을 "가로현수막(낮)", "간이의자(낮)", "간이테이블(낮)" 중 선택하여 수정
 - python model_test.py로 실행 => ./results/{dataset_name} 아래에 모형 출력 결과가 저장됨

- 학습 모델 리스트
 - 가로현수막(낮)
 - 간이의자(낮)
 - 간이테이블(낮)
 - 간이표지판(낮)
 - 골목비포장쓰레기(낮)
 - 도로변비포장쓰레기(낮)
 - 도로변쓰레기봉투더미(낮)
 - 비활성에어간판(낮)
 - 세로현수막(낮)
 - 안전간판(낮)
 - 엑스배너(낮)
 - 중장비(낮)
 - 활성에어간판(낮)
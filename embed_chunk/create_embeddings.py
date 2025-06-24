from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import gc
from tqdm import tqdm

def create_embeddings():
    print("임베딩 생성 시작...")
    # 임베딩 모델 로드
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v1"
    )

    texts = []
    metadatas = []

    # chunks 폴더 내 모든 txt 파일 처리
    files = [f for f in os.listdir("chunks") if f.endswith(".txt")]
    print(f"총 {len(files)}개 파일 처리 예정")
    for filename in tqdm(files, desc="파일 처리 중"):
        with open(os.path.join("chunks", filename), "r", encoding="utf-8") as f:
            content = f.read()
            texts.append(content)
            metadatas.append({"source": filename})
        gc.collect()

    print(f"총 {len(texts)}개 문서 임베딩")
    # 벡터 DB 생성
    db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    # 저장
    db.save_local("embed_faiss")
    print("임베딩 생성 및 저장 완료!")

if __name__ == "__main__":
    create_embeddings() 
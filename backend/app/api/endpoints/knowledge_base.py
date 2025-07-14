from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import json

from ...core.database import get_db
from ...services.document_ingestion import DocumentIngestionService
from ...schemas.client import KnowledgeBaseCreateRequest, KnowledgeBaseResponse
from ...models.client import KnowledgeBase, Client

router = APIRouter()


@router.post("/", response_model=KnowledgeBaseResponse)
async def create_knowledge_base(
    kb_request: KnowledgeBaseCreateRequest,
    client_id: str,
    db: Session = Depends(get_db)
):
    """Create a new knowledge base for a client"""
    
    # Verify client exists
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Create knowledge base
    kb = KnowledgeBase(
        client_id=client_id,
        name=kb_request.name,
        description=kb_request.description,
        chunking_config=kb_request.config.dict(),
        embedding_config={"model": kb_request.config.embedding_model}
    )
    
    db.add(kb)
    db.commit()
    db.refresh(kb)
    
    return kb


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    kb_id: int,
    db: Session = Depends(get_db)
):
    """Get knowledge base details"""
    
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    return kb


@router.get("/client/{client_id}", response_model=List[KnowledgeBaseResponse])
async def list_client_knowledge_bases(
    client_id: str,
    db: Session = Depends(get_db)
):
    """List all knowledge bases for a client"""
    
    kbs = db.query(KnowledgeBase).filter(KnowledgeBase.client_id == client_id).all()
    return kbs


@router.post("/{kb_id}/documents/upload")
async def upload_document(
    kb_id: int,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    metadata: Optional[str] = Form("{}"),
    source_url: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload a single document to a knowledge base"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Parse metadata
    try:
        metadata_dict = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")
    
    # Read file data
    file_data = await file.read()
    
    # Initialize ingestion service
    ingestion_service = DocumentIngestionService(db)
    
    try:
        document = await ingestion_service.ingest_document(
            knowledge_base_id=kb_id,
            file_data=file_data,
            filename=file.filename,
            title=title,
            metadata=metadata_dict,
            source_url=source_url
        )
        
        return {
            "document_id": document.id,
            "title": document.title,
            "status": document.processing_status,
            "content_length": len(document.content),
            "content_type": document.content_type,
            "metadata": document.metadata
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{kb_id}/documents/upload-multiple")
async def upload_multiple_documents(
    kb_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Upload multiple documents to a knowledge base"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    # Prepare file data
    file_data_list = []
    for file in files:
        data = await file.read()
        file_data_list.append({
            'data': data,
            'filename': file.filename,
            'title': file.filename
        })
    
    # Initialize ingestion service
    ingestion_service = DocumentIngestionService(db)
    
    try:
        documents = await ingestion_service.ingest_multiple_documents(
            knowledge_base_id=kb_id,
            files=file_data_list
        )
        
        return {
            "uploaded": len(documents),
            "documents": [
                {
                    "document_id": doc.id,
                    "title": doc.title,
                    "status": doc.processing_status,
                    "content_length": len(doc.content)
                }
                for doc in documents
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{kb_id}/documents")
async def list_documents(
    kb_id: int,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List documents in a knowledge base"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    from ...models.client import Document
    
    documents = db.query(Document).filter(
        Document.knowledge_base_id == kb_id
    ).offset(skip).limit(limit).all()
    
    return {
        "documents": [
            {
                "id": doc.id,
                "title": doc.title,
                "content_type": doc.content_type,
                "processing_status": doc.processing_status,
                "content_length": len(doc.content),
                "chunks_count": len(doc.chunks),
                "created_at": doc.created_at,
                "updated_at": doc.updated_at
            }
            for doc in documents
        ],
        "total": len(documents),
        "skip": skip,
        "limit": limit
    }


@router.get("/{kb_id}/documents/{doc_id}")
async def get_document(
    kb_id: int,
    doc_id: int,
    db: Session = Depends(get_db)
):
    """Get document details"""
    
    from ...models.client import Document
    
    document = db.query(Document).filter(
        Document.id == doc_id,
        Document.knowledge_base_id == kb_id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "id": document.id,
        "title": document.title,
        "content": document.content,
        "content_type": document.content_type,
        "source_url": document.source_url,
        "processing_status": document.processing_status,
        "error_message": document.error_message,
        "metadata": document.metadata,
        "content_hash": document.content_hash,
        "version": document.version,
        "chunks_count": len(document.chunks),
        "created_at": document.created_at,
        "updated_at": document.updated_at
    }


@router.put("/{kb_id}/documents/{doc_id}")
async def update_document(
    kb_id: int,
    doc_id: int,
    file: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    metadata: Optional[str] = Form("{}"),
    source_url: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Update a document"""
    
    # Parse metadata
    try:
        metadata_dict = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")
    
    # Read file data if provided
    file_data = None
    if file:
        file_data = await file.read()
    
    # Initialize ingestion service
    ingestion_service = DocumentIngestionService(db)
    
    try:
        document = await ingestion_service.update_document(
            document_id=doc_id,
            file_data=file_data,
            title=title,
            metadata=metadata_dict,
            source_url=source_url
        )
        
        return {
            "document_id": document.id,
            "title": document.title,
            "status": document.processing_status,
            "content_length": len(document.content),
            "version": document.version
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{kb_id}/documents/{doc_id}")
async def delete_document(
    kb_id: int,
    doc_id: int,
    db: Session = Depends(get_db)
):
    """Delete a document"""
    
    ingestion_service = DocumentIngestionService(db)
    
    success = await ingestion_service.delete_document(doc_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": "Document deleted successfully"}


@router.get("/{kb_id}/documents/{doc_id}/status")
async def get_document_status(
    kb_id: int,
    doc_id: int,
    db: Session = Depends(get_db)
):
    """Get document processing status"""
    
    ingestion_service = DocumentIngestionService(db)
    
    status = ingestion_service.get_document_status(doc_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return status


@router.get("/supported-file-types")
async def get_supported_file_types():
    """Get list of supported file types for upload"""
    
    from ...services.document_ingestion import DocumentProcessor
    
    processor = DocumentProcessor()
    supported_types = processor.get_supported_file_types()
    
    return {
        "supported_types": supported_types,
        "extensions": list(processor.SUPPORTED_TYPES.values())
    }
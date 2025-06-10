from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import json
from datetime import datetime
from decimal import Decimal

app = FastAPI(
    title="Financial Transaction Vector Store API", 
    description="Store financial transaction data for RAG-powered LLM queries"
)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path="./financial_vector_db")

collection = chroma_client.get_or_create_collection(
    name="financial_transactions",
    metadata={"description": "Financial transaction data for RAG retrieval"}
)

class Transaction(BaseModel):
    date: str = Field(..., description="Transaction date in YYYY-MM-DD format")
    description: str = Field(..., description="Transaction description")
    type: str = Field(..., description="Transaction type (e.g., debit, credit, transfer)")
    amount: float = Field(..., description="Transaction amount (negative for expenses)")

class FinancialData(BaseModel):
    initial_balance: float = Field(..., description="Starting account balance")
    transactions: List[Transaction] = Field(..., description="List of transactions")
    account_id: Optional[str] = Field(None, description="Account identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class FinancialDataResponse(BaseModel):
    document_id: str
    message: str
    summary: Dict[str, Any]

class FinancialSearchQuery(BaseModel):
    query: str = Field(..., description="Natural language query about transactions")
    n_results: Optional[int] = Field(5, description="Number of results to return")
    date_filter: Optional[str] = Field(None, description="Filter by date range (YYYY-MM-DD)")
    amount_filter: Optional[Dict[str, float]] = Field(None, description="Filter by amount range")

class FinancialSearchResult(BaseModel):
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]

def create_financial_narrative(data: FinancialData) -> str:
    total_transactions = len(data.transactions)
    total_spent = sum(abs(t.amount) for t in data.transactions if t.amount < 0)
    total_received = sum(t.amount for t in data.transactions if t.amount > 0)
    final_balance = data.initial_balance + sum(t.amount for t in data.transactions)
    transaction_types = {}
    monthly_summary = {}
    for transaction in data.transactions:
        if transaction.type not in transaction_types:
            transaction_types[transaction.type] = []
        transaction_types[transaction.type].append(transaction)
        month = transaction.date[:7]
        if month not in monthly_summary:
            monthly_summary[month] = {"count": 0, "total": 0}
        monthly_summary[month]["count"] += 1
        monthly_summary[month]["total"] += transaction.amount
    narrative_parts = [
        f"Financial Account Summary: Starting balance of ${data.initial_balance:.2f}",
        f"Total transactions: {total_transactions}",
        f"Total money spent: ${total_spent:.2f}",
        f"Total money received: ${total_received:.2f}",
        f"Final calculated balance: ${final_balance:.2f}"
    ]
    for tx_type, transactions in transaction_types.items():
        avg_amount = sum(t.amount for t in transactions) / len(transactions)
        narrative_parts.append(
            f"{tx_type.title()} transactions: {len(transactions)} occurrences, "
            f"average amount ${avg_amount:.2f}"
        )
    for month, summary in monthly_summary.items():
        narrative_parts.append(
            f"Month {month}: {summary['count']} transactions totaling ${summary['total']:.2f}"
        )
    narrative_parts.append("Individual transactions:")
    for i, transaction in enumerate(data.transactions):
        amount_desc = "expense" if transaction.amount < 0 else "income"
        narrative_parts.append(
            f"Transaction {i+1}: {transaction.date} - {transaction.description} "
            f"({transaction.type}) ${abs(transaction.amount):.2f} {amount_desc}"
        )
    return " | ".join(narrative_parts)

def extract_financial_metadata(data: FinancialData) -> Dict[str, Any]:
    transactions = data.transactions
    total_spent = sum(abs(t.amount) for t in transactions if t.amount < 0)
    total_received = sum(t.amount for t in transactions if t.amount > 0)
    final_balance = data.initial_balance + sum(t.amount for t in transactions)
    transaction_types = list(set(t.type for t in transactions))
    date_range = {
        "earliest": min(t.date for t in transactions) if transactions else None,
        "latest": max(t.date for t in transactions) if transactions else None
    }
    return {
        "account_id": data.account_id,
        "initial_balance": data.initial_balance,
        "final_balance": final_balance,
        "transaction_count": len(transactions),
        "total_spent": total_spent,
        "total_received": total_received,
        "net_change": total_received - total_spent,
        "transaction_types": ", ".join(transaction_types),
        "date_range": json.dumps(date_range),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/store-financial-data", response_model=FinancialDataResponse)
async def store_financial_data(data: FinancialData):
    try:
        doc_id = data.account_id or str(uuid.uuid4())
        narrative_text = create_financial_narrative(data)
        embedding = embedding_model.encode(narrative_text).tolist()
        metadata = extract_financial_metadata(data)
        if data.metadata:
            metadata.update(data.metadata)
        metadata["original_data"] = data.model_dump_json()
        metadata["document_type"] = "financial_transactions"
        collection.upsert(
            documents=[narrative_text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
        return FinancialDataResponse(
            document_id=doc_id,
            message="Financial data successfully stored for RAG retrieval",
            summary={
                "transaction_count": metadata["transaction_count"],
                "final_balance": metadata["final_balance"],
                "date_range": metadata["date_range"]
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing financial data: {str(e)}")

@app.post("/search-financial", response_model=FinancialSearchResult)
async def search_financial_data(query: FinancialSearchQuery):
    try:
        query_embedding = embedding_model.encode(query.query).tolist()
        where_clause = {"document_type": "financial_transactions"}
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=query.n_results,
            where=where_clause,
            include=['documents', 'metadatas', 'distances']
        )
        processed_results = []
        total_balance = 0
        total_transactions = 0
        for i, doc_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            if 'original_data' in metadata:
                original_data = json.loads(metadata['original_data'])
                processed_results.append({
                    "document_id": doc_id,
                    "relevance_score": 1 - results['distances'][0][i],
                    "financial_data": original_data,
                    "summary": {
                        "initial_balance": metadata.get("initial_balance"),
                        "final_balance": metadata.get("final_balance"),
                        "transaction_count": metadata.get("transaction_count"),
                        "date_range": metadata.get("date_range")
                    },
                    "narrative": results['documents'][0][i]
                })
                total_balance += metadata.get("final_balance", 0)
                total_transactions += metadata.get("transaction_count", 0)
        return FinancialSearchResult(
            results=processed_results,
            summary={
                "query": query.query,
                "total_accounts_found": len(processed_results),
                "combined_balance": total_balance,
                "total_transactions": total_transactions
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching financial data: {str(e)}")

@app.get("/financial-summary")
async def get_financial_summary():
    try:
        all_docs = collection.get(
            where={"document_type": "financial_transactions"},
            include=['metadatas']
        )
        if not all_docs['ids']:
            return {"message": "No financial data found"}
        total_accounts = len(all_docs['ids'])
        total_balance = sum(meta.get("final_balance", 0) for meta in all_docs['metadatas'])
        total_transactions = sum(meta.get("transaction_count", 0) for meta in all_docs['metadatas'])
        all_types = set()
        for meta in all_docs['metadatas']:
            if meta.get("transaction_types"):
                all_types.update(meta["transaction_types"])
        return {
            "total_accounts": total_accounts,
            "combined_balance": total_balance,
            "total_transactions": total_transactions,
            "transaction_types": list(all_types),
            "average_balance_per_account": total_balance / total_accounts if total_accounts > 0 else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting financial summary: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        doc_count = collection.count()
        return {
            "status": "healthy",
            "total_documents": doc_count,
            "embedding_model": "all-MiniLM-L6-v2",
            "database_type": "ChromaDB",
            "optimized_for": "financial_transaction_rag"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
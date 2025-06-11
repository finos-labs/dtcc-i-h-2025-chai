from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import json
from datetime import datetime, timedelta
from decimal import Decimal
from collections import defaultdict
import statistics
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Financial Transaction Vector Store API", 
    description="Store financial transaction data for RAG-powered LLM queries"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    date_filter: Optional[str] = Field(None, description="Filter by specific date (YYYY-MM-DD) or date range")
    amount_filter: Optional[Dict[str, float]] = Field(None, description="Filter by amount range")

class FinancialSearchResult(BaseModel):
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]

class FinancialSummaryQuery(BaseModel):
    account_id: Optional[str] = Field(None, description="Specific account ID to analyze")
    analysis_type: Optional[str] = Field("comprehensive", description="Type of analysis: comprehensive, spending, income, trends")
    date_range_days: Optional[int] = Field(None, description="Number of days to look back for analysis")
    include_predictions: Optional[bool] = Field(True, description="Include spending predictions and trends")

class FinancialSummaryResponse(BaseModel):
    summary_type: str
    account_count: int
    analysis_period: Dict[str, Any]
    financial_health: Dict[str, Any]
    spending_analysis: Dict[str, Any]
    income_analysis: Dict[str, Any]
    trends: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]

class AllRecordsQuery(BaseModel):
    include_documents: Optional[bool] = Field(True, description="Include full document narratives")
    include_metadata: Optional[bool] = Field(True, description="Include metadata")
    include_original_data: Optional[bool] = Field(False, description="Include original transaction data")
    limit: Optional[int] = Field(None, description="Limit number of records returned")
    account_id_filter: Optional[str] = Field(None, description="Filter by specific account ID")

class AllRecordsResponse(BaseModel):
    total_records: int
    records_returned: int
    records: List[Dict[str, Any]]
    summary: Dict[str, Any]


def filter_transactions_by_date(transactions: List[Dict], date_filter: str) -> List[Dict]:
    """Filter transactions by date or date range"""
    if not date_filter:
        return transactions
    
    filtered_transactions = []
    
    # Check if it's a date range (contains 'to' or '-' between dates)
    if ' to ' in date_filter.lower():
        parts = date_filter.lower().split(' to ')
        if len(parts) == 2:
            start_date = parts[0].strip()
            end_date = parts[1].strip()
            for txn in transactions:
                if start_date <= txn['date'] <= end_date:
                    filtered_transactions.append(txn)
    elif '--' in date_filter:  # Range with double dash
        parts = date_filter.split('--')
        if len(parts) == 2:
            start_date = parts[0].strip()
            end_date = parts[1].strip()
            for txn in transactions:
                if start_date <= txn['date'] <= end_date:
                    filtered_transactions.append(txn)
    else:
        # Single date filter
        target_date = date_filter.strip()
        for txn in transactions:
            if txn['date'] == target_date:
                filtered_transactions.append(txn)
    
    return filtered_transactions

def filter_transactions_by_amount(transactions: List[Dict], amount_filter: Dict[str, float]) -> List[Dict]:
    """Filter transactions by amount range"""
    if not amount_filter:
        return transactions
    
    filtered_transactions = []
    min_amount = amount_filter.get('min', float('-inf'))
    max_amount = amount_filter.get('max', float('inf'))
    
    for txn in transactions:
        if min_amount <= abs(txn['amount']) <= max_amount:
            filtered_transactions.append(txn)
    
    return filtered_transactions

def create_financial_narrative(data: FinancialData) -> str:
    total_transactions = len(data.transactions)
    total_spent = sum(abs(t.amount) for t in data.transactions if t.amount < 0)
    total_received = sum(t.amount for t in data.transactions if t.amount > 0)
    final_balance = data.initial_balance + sum(t.amount for t in data.transactions)
    transaction_types = {}
    monthly_summary = {}
    daily_summary = {}
    
    for transaction in data.transactions:
        # Transaction types
        if transaction.type not in transaction_types:
            transaction_types[transaction.type] = []
        transaction_types[transaction.type].append(transaction)
        
        # Monthly summary
        month = transaction.date[:7]
        if month not in monthly_summary:
            monthly_summary[month] = {"count": 0, "total": 0}
        monthly_summary[month]["count"] += 1
        monthly_summary[month]["total"] += transaction.amount
        
        # Daily summary for better date-based searches
        date = transaction.date
        if date not in daily_summary:
            daily_summary[date] = {"count": 0, "total": 0, "transactions": []}
        daily_summary[date]["count"] += 1
        daily_summary[date]["total"] += transaction.amount
        daily_summary[date]["transactions"].append({
            "description": transaction.description,
            "type": transaction.type,
            "amount": transaction.amount
        })
    
    narrative_parts = [
        f"Financial Account Summary: Starting balance of ${data.initial_balance:.2f}",
        f"Total transactions: {total_transactions}",
        f"Total money spent: ${total_spent:.2f}",
        f"Total money received: ${total_received:.2f}",
        f"Final calculated balance: ${final_balance:.2f}"
    ]
    
    # Add transaction type summaries
    for tx_type, transactions in transaction_types.items():
        avg_amount = sum(t.amount for t in transactions) / len(transactions)
        narrative_parts.append(
            f"{tx_type.title()} transactions: {len(transactions)} occurrences, "
            f"average amount ${avg_amount:.2f}"
        )
    
    # Add monthly summaries
    for month, summary in monthly_summary.items():
        narrative_parts.append(
            f"Month {month}: {summary['count']} transactions totaling ${summary['total']:.2f}"
        )
    
    # Add daily summaries for better date-based retrieval
    for date, summary in daily_summary.items():
        transaction_details = []
        for txn in summary["transactions"]:
            amount_desc = "expense" if txn["amount"] < 0 else "income"
            transaction_details.append(
                f"{txn['description']} ({txn['type']}) ${abs(txn['amount']):.2f} {amount_desc}"
            )
        narrative_parts.append(
            f"Date {date}: {summary['count']} transactions totaling ${summary['total']:.2f} - "
            f"Details: {'; '.join(transaction_details)}"
        )
    
    # Add individual transaction details
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
    
    # Enhanced date tracking
    dates = [t.date for t in transactions]
    date_range = {
        "earliest": min(dates) if dates else None,
        "latest": max(dates) if dates else None,
        "all_dates": list(set(dates))  # Store all unique dates for filtering
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

def analyze_spending_patterns(all_transactions: List[Dict]) -> Dict[str, Any]:
    """Analyze spending patterns using ML embeddings for categorization"""
    
    # Group transactions by type and description similarity
    expense_transactions = [t for t in all_transactions if t['amount'] < 0]
    income_transactions = [t for t in all_transactions if t['amount'] >= 0]
    
    # Category analysis using embedding similarity
    categories = defaultdict(list)
    for txn in expense_transactions:
        # Simple categorization based on description keywords
        desc_lower = txn['description'].lower()
        if any(word in desc_lower for word in ['grocery', 'food', 'restaurant', 'cafe']):
            categories['Food & Dining'].append(abs(txn['amount']))
        elif any(word in desc_lower for word in ['gas', 'fuel', 'transport', 'uber', 'taxi']):
            categories['Transportation'].append(abs(txn['amount']))
        elif any(word in desc_lower for word in ['shop', 'store', 'amazon', 'purchase']):
            categories['Shopping'].append(abs(txn['amount']))
        elif any(word in desc_lower for word in ['bill', 'utility', 'electric', 'water', 'internet']):
            categories['Bills & Utilities'].append(abs(txn['amount']))
        else:
            categories['Other'].append(abs(txn['amount']))
    
    # Calculate category statistics
    category_stats = {}
    for category, amounts in categories.items():
        if amounts:
            category_stats[category] = {
                'total': round(sum(amounts), 2),
                'average': round(statistics.mean(amounts), 2),
                'count': len(amounts),
                'percentage': 0  # Will calculate after getting total
            }
    
    total_expenses = sum(abs(t['amount']) for t in expense_transactions)
    for category in category_stats:
        category_stats[category]['percentage'] = round(
            (category_stats[category]['total'] / total_expenses) * 100, 1
        ) if total_expenses > 0 else 0
    
    return {
        'total_expenses': round(total_expenses, 2),
        'total_income': round(sum(t['amount'] for t in income_transactions), 2),
        'expense_count': len(expense_transactions),
        'income_count': len(income_transactions),
        'categories': category_stats,
        'average_expense': round(statistics.mean([abs(t['amount']) for t in expense_transactions]), 2) if expense_transactions else 0,
        'largest_expense': max([abs(t['amount']) for t in expense_transactions]) if expense_transactions else 0
    }

def generate_financial_insights(spending_analysis: Dict, trends: Dict, financial_health: Dict) -> List[str]:
    """Generate AI-powered financial insights"""
    insights = []
    
    # Spending insights
    if spending_analysis['categories']:
        top_category = max(spending_analysis['categories'].items(), key=lambda x: x[1]['total'])
        insights.append(f"Your largest spending category is {top_category[0]} at ${top_category[1]['total']:.2f} ({top_category[1]['percentage']:.1f}% of expenses)")
    
    # Balance trend insights
    if trends.get('balance_trend') == 'increasing':
        insights.append("Your account balance is trending upward, indicating positive financial momentum")
    elif trends.get('balance_trend') == 'decreasing':
        insights.append("Your account balance is declining - consider reviewing your spending habits")
    
    # Cash flow insights
    net_flow = spending_analysis['total_income'] - spending_analysis['total_expenses']
    if net_flow > 0:
        insights.append(f"You have a positive cash flow of ${net_flow:.2f} this period")
    else:
        insights.append(f"You have a negative cash flow of ${abs(net_flow):.2f} - expenses exceed income")
    
    # Frequency insights
    if spending_analysis['expense_count'] > 0:
        avg_daily_expenses = spending_analysis['expense_count'] / max(trends.get('analysis_days', 30), 1)
        if avg_daily_expenses > 5:
            insights.append(f"You average {avg_daily_expenses:.1f} transactions per day - consider consolidating purchases")
    
    return insights

def generate_recommendations(spending_analysis: Dict, insights: List[str]) -> List[str]:
    """Generate personalized financial recommendations"""
    recommendations = []
    
    # Category-based recommendations
    if spending_analysis['categories']:
        top_categories = sorted(spending_analysis['categories'].items(), 
                              key=lambda x: x[1]['total'], reverse=True)[:3]
        
        for category, stats in top_categories:
            if stats['percentage'] > 30:
                recommendations.append(f"Consider reducing {category} spending - it represents {stats['percentage']:.1f}% of your expenses")
    
    # Cash flow recommendations
    income_expense_ratio = spending_analysis['total_income'] / max(spending_analysis['total_expenses'], 1)
    if income_expense_ratio < 1.2:
        recommendations.append("Your expenses are close to your income - consider building an emergency fund")
    
    # Frequency recommendations
    if spending_analysis['average_expense'] < 50 and spending_analysis['expense_count'] > 20:
        recommendations.append("You have many small transactions - consider budgeting for larger, planned purchases")
    
    # Savings recommendations
    if spending_analysis['total_income'] > spending_analysis['total_expenses']:
        savings_potential = spending_analysis['total_income'] - spending_analysis['total_expenses']
        recommendations.append(f"You could potentially save ${savings_potential:.2f} by maintaining current spending levels")
    
    return recommendations

@app.get("/all-records", response_model=AllRecordsResponse)
async def get_all_records(
    include_documents: bool = True,
    include_metadata: bool = True, 
    include_original_data: bool = False,
    limit: Optional[int] = None,
    account_id_filter: Optional[str] = None
):
    """
    Retrieve all financial records from the vector store
    
    Args:
        include_documents: Include the full narrative documents
        include_metadata: Include record metadata
        include_original_data: Include the original transaction data
        limit: Maximum number of records to return
        account_id_filter: Filter by specific account ID
    """
    try:
        # Build where clause for filtering
        where_clause = {"document_type": "financial_transactions"}
        if account_id_filter:
            where_clause["account_id"] = account_id_filter
        
        # Get total count first
        total_count = collection.count()
        
        # Determine what to include in the query
        include_list = []
        if include_documents:
            include_list.append('documents')
        if include_metadata:
            include_list.append('metadatas')
        if 'distances' not in include_list:
            include_list.append('distances')
        # Remove 'ids' from include_list (not a valid option for ChromaDB)
        
        # Query all records (ChromaDB doesn't have a direct "get all" method, so we use a large limit)
        query_limit = limit if limit else 1000  # Default reasonable limit
        
        # Use a generic embedding query to get all financial records
        generic_embedding = embedding_model.encode("financial transactions").tolist()
        
        results = collection.query(
            query_embeddings=[generic_embedding],
            n_results=query_limit,
            where=where_clause,
            include=include_list
        )
        
        # Process results
        processed_records = []
        account_summary = defaultdict(lambda: {
            'transaction_count': 0,
            'total_balance': 0,
            'date_range': {'earliest': None, 'latest': None}
        })
        
        for i, doc_id in enumerate(results['ids'][0]):
            record = {
                "document_id": doc_id,
                "relevance_score": 1 - results['distances'][0][i] if 'distances' in results else None
            }
            
            # Add documents if requested
            if include_documents and 'documents' in results:
                record["narrative"] = results['documents'][0][i]
            
            # Add metadata if requested
            if include_metadata and 'metadatas' in results:
                metadata = results['metadatas'][0][i]
                record["metadata"] = {
                    "account_id": metadata.get("account_id"),
                    "initial_balance": metadata.get("initial_balance"),
                    "final_balance": metadata.get("final_balance"),
                    "transaction_count": metadata.get("transaction_count"),
                    "total_spent": metadata.get("total_spent"),
                    "total_received": metadata.get("total_received"),
                    "transaction_types": metadata.get("transaction_types"),
                    "date_range": metadata.get("date_range"),
                    "timestamp": metadata.get("timestamp")
                }
                
                # Update account summary
                account_id = metadata.get("account_id", "unknown")
                account_summary[account_id]['transaction_count'] += metadata.get("transaction_count", 0)
                account_summary[account_id]['total_balance'] += metadata.get("final_balance", 0)
                
                # Update date range
                if metadata.get("date_range"):
                    try:
                        date_range = json.loads(metadata["date_range"])
                        earliest = date_range.get("earliest")
                        latest = date_range.get("latest")
                        
                        if earliest:
                            if not account_summary[account_id]['date_range']['earliest'] or earliest < account_summary[account_id]['date_range']['earliest']:
                                account_summary[account_id]['date_range']['earliest'] = earliest
                        
                        if latest:
                            if not account_summary[account_id]['date_range']['latest'] or latest > account_summary[account_id]['date_range']['latest']:
                                account_summary[account_id]['date_range']['latest'] = latest
                    except json.JSONDecodeError:
                        pass
            
            # Add original data if requested
            if include_original_data and include_metadata and 'metadatas' in results:
                metadata = results['metadatas'][0][i]
                if 'original_data' in metadata:
                    try:
                        record["original_data"] = json.loads(metadata['original_data'])
                    except json.JSONDecodeError:
                        record["original_data"] = None
            
            processed_records.append(record)
        
        # Create summary statistics
        summary = {
            "unique_accounts": len(account_summary),
            "account_summary": dict(account_summary),
            "total_transactions": sum(acc['transaction_count'] for acc in account_summary.values()),
            "total_balance_across_accounts": round(sum(acc['total_balance'] for acc in account_summary.values()), 2),
            "query_parameters": {
                "include_documents": include_documents,
                "include_metadata": include_metadata,
                "include_original_data": include_original_data,
                "limit_applied": limit,
                "account_filter": account_id_filter
            }
        }
        
        # Add date range summary
        all_earliest_dates = [acc['date_range']['earliest'] for acc in account_summary.values() if acc['date_range']['earliest']]
        all_latest_dates = [acc['date_range']['latest'] for acc in account_summary.values() if acc['date_range']['latest']]
        
        if all_earliest_dates and all_latest_dates:
            summary["overall_date_range"] = {
                "earliest": min(all_earliest_dates),
                "latest": max(all_latest_dates)
            }
        
        return AllRecordsResponse(
            total_records=total_count,
            records_returned=len(processed_records),
            records=processed_records,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving all records: {str(e)}")

@app.post("/financial-summary", response_model=FinancialSummaryResponse)
async def get_financial_summary(query: FinancialSummaryQuery):
    """Generate comprehensive financial summary using ML analysis"""
    try:
        # Build query to retrieve relevant financial data
        where_clause = {"document_type": "financial_transactions"}
        if query.account_id:
            where_clause["account_id"] = query.account_id
        
        # Get all relevant documents
        results = collection.query(
            query_embeddings=[embedding_model.encode("financial summary analysis").tolist()],
            n_results=100,  # Get more results for comprehensive analysis
            where=where_clause,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['ids'][0]:
            raise HTTPException(status_code=404, detail="No financial data found")
        
        # Aggregate all transaction data
        all_transactions = []
        all_accounts = []
        total_initial_balance = 0
        total_final_balance = 0
        earliest_date = None
        latest_date = None
        
        for i, doc_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            if 'original_data' in metadata:
                original_data = json.loads(metadata['original_data'])
                all_accounts.append(original_data)
                all_transactions.extend(original_data['transactions'])
                
                total_initial_balance += original_data['initial_balance']
                total_final_balance += metadata.get('final_balance', 0)
                
                # Track date range
                date_range = json.loads(metadata.get('date_range', '{}'))
                if date_range.get('earliest'):
                    if not earliest_date or date_range['earliest'] < earliest_date:
                        earliest_date = date_range['earliest']
                if date_range.get('latest'):
                    if not latest_date or date_range['latest'] > latest_date:
                        latest_date = date_range['latest']
        
        # Filter by date range if specified
        if query.date_range_days:
            cutoff_date = (datetime.now() - timedelta(days=query.date_range_days)).strftime('%Y-%m-%d')
            all_transactions = [t for t in all_transactions if t['date'] >= cutoff_date]
        
        # Perform comprehensive analysis
        spending_analysis = analyze_spending_patterns(all_transactions)
        
        # Calculate trends
        monthly_data = defaultdict(lambda: {'income': 0, 'expenses': 0, 'net': 0})
        for txn in all_transactions:
            month = txn['date'][:7]  # YYYY-MM
            if txn['amount'] >= 0:
                monthly_data[month]['income'] += txn['amount']
            else:
                monthly_data[month]['expenses'] += abs(txn['amount'])
            monthly_data[month]['net'] += txn['amount']
        
        # Determine trends
        monthly_nets = list(monthly_data.values())
        balance_trend = "stable"
        if len(monthly_nets) >= 2:
            recent_avg = statistics.mean([m['net'] for m in monthly_nets[-2:]])
            earlier_avg = statistics.mean([m['net'] for m in monthly_nets[:-2]]) if len(monthly_nets) > 2 else monthly_nets[0]['net']
            if recent_avg > earlier_avg * 1.1:
                balance_trend = "increasing"
            elif recent_avg < earlier_avg * 0.9:
                balance_trend = "decreasing"
        
        trends = {
            'balance_trend': balance_trend,
            'monthly_data': dict(monthly_data),
            'analysis_days': query.date_range_days or (
                (datetime.now() - datetime.strptime(earliest_date, '%Y-%m-%d')).days 
                if earliest_date else 30
            )
        }
        
        # Financial health metrics
        financial_health = {
            'total_balance': round(total_final_balance, 2),
            'net_worth_change': round(total_final_balance - total_initial_balance, 2),
            'expense_to_income_ratio': round(
                spending_analysis['total_expenses'] / max(spending_analysis['total_income'], 1), 2
            ),
            'financial_stability': 'good' if total_final_balance > total_initial_balance else 'needs_attention'
        }
        
        # Generate AI insights and recommendations
        insights = generate_financial_insights(spending_analysis, trends, financial_health)
        recommendations = generate_recommendations(spending_analysis, insights)
        
        return FinancialSummaryResponse(
            summary_type=query.analysis_type,
            account_count=len(all_accounts),
            analysis_period={
                'start_date': earliest_date,
                'end_date': latest_date,
                'days_analyzed': trends['analysis_days']
            },
            financial_health=financial_health,
            spending_analysis=spending_analysis,
            income_analysis={
                'total_income': spending_analysis['total_income'],
                'income_transactions': spending_analysis['income_count'],
                'average_income_per_transaction': round(
                    spending_analysis['total_income'] / max(spending_analysis['income_count'], 1), 2
                )
            },
            trends=trends,
            insights=insights,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating financial summary: {str(e)}")

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
        
        # Get all relevant documents first
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=query.n_results * 2,  # Get more results to filter from
            where=where_clause,
            include=['documents', 'metadatas', 'distances']
        )
        
        processed_results = []
        total_balance = 0
        total_transactions = 0
        total_transfer_expense = 0.0
        filtered_transaction_count = 0
        matching_transactions = []

        for i, doc_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            if 'original_data' in metadata:
                original_data = json.loads(metadata['original_data'])
                transactions = original_data["transactions"]
                
                # Apply date filter if specified
                if query.date_filter:
                    transactions = filter_transactions_by_date(transactions, query.date_filter)
                
                # Apply amount filter if specified
                if query.amount_filter:
                    transactions = filter_transactions_by_amount(transactions, query.amount_filter)
                
                # If after filtering we have no transactions, skip this document
                if not transactions and (query.date_filter or query.amount_filter):
                    continue
                
                # Process transfer expenses
                if "transfer" in query.query.lower():
                    for txn in transactions:
                        if "transfer" in txn["type"].lower() and txn["amount"] < 0:
                            total_transfer_expense += abs(txn["amount"])

                # Count filtered transactions
                filtered_transaction_count += len(transactions)
                matching_transactions.extend(transactions)

                filtered_data = original_data.copy()
                filtered_data["transactions"] = transactions
                
                filtered_final_balance = original_data["initial_balance"] + sum(t["amount"] for t in transactions)

                processed_results.append({
                    "document_id": doc_id,
                    "relevance_score": 1 - results['distances'][0][i],
                    "financial_data": filtered_data,
                    "summary": {
                        "initial_balance": metadata.get("initial_balance"),
                        "final_balance": filtered_final_balance,
                        "transaction_count": len(transactions),
                        "date_range": metadata.get("date_range"),
                        "filtered": bool(query.date_filter or query.amount_filter)
                    },
                    "narrative": results['documents'][0][i]
                })

                total_balance += filtered_final_balance
                total_transactions += len(transactions)

        processed_results = processed_results[:query.n_results]

        summary = {
            "query": query.query,
            "total_accounts_found": len(processed_results),
            "combined_balance": round(total_balance, 2),
            "total_transactions": total_transactions,
            "date_filter_applied": query.date_filter is not None,
            "amount_filter_applied": query.amount_filter is not None,
        }

        if query.date_filter:
            summary["filtered_by_date"] = query.date_filter
            summary["transactions_on_date"] = filtered_transaction_count
            
            if filtered_transaction_count == 0:
                summary["message"] = f"No transactions found for date: {query.date_filter}"
            else:
                date_breakdown = defaultdict(int)
                for txn in matching_transactions:
                    date_breakdown[txn['date']] += 1
                summary["date_breakdown"] = dict(date_breakdown)

        if "transfer" in query.query.lower():
            summary["total_transfer_expense"] = round(total_transfer_expense, 2)

        return FinancialSearchResult(results=processed_results, summary=summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching financial data: {str(e)}")

@app.get("/debug-dates/{account_id}")
async def debug_account_dates(account_id: str):
    """Debug endpoint to check what dates exist for a specific account"""
    try:
        results = collection.query(
            query_embeddings=[embedding_model.encode("debug dates").tolist()],
            n_results=10,
            where={"document_type": "financial_transactions", "account_id": account_id},
            include=['documents', 'metadatas']
        )
        
        if not results['ids'][0]:
            return {"error": "Account not found", "account_id": account_id}
        
        dates_info = []
        for i, doc_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            if 'original_data' in metadata:
                original_data = json.loads(metadata['original_data'])
                transactions = original_data['transactions']
                dates = [t['date'] for t in transactions]
                dates_info.append({
                    "document_id": doc_id,
                    "transaction_dates": sorted(list(set(dates))),
                    "transaction_count": len(transactions),
                    "date_range": json.loads(metadata.get('date_range', '{}'))
                })
        
        return {
            "account_id": account_id,
            "documents_found": len(dates_info),
            "dates_info": dates_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error debugging dates: {str(e)}")

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
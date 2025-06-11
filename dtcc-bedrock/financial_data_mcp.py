from mcp.server.fastmcp import FastMCP
import requests
import json
from typing import Optional, Dict, Any, List
from datetime import datetime

mcp = FastMCP("FinancialVectorDB")

# Configuration for the Financial Vector DB API
FINANCIAL_API_BASE_URL = "http://localhost:8000"  # Update this to match your FastAPI server

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
    """Make a request to the Financial Vector DB API"""
    url = f"{FINANCIAL_API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            return {"error": f"Unsupported HTTP method: {method}"}
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.ConnectionError:
        return {"error": f"Could not connect to Financial API at {FINANCIAL_API_BASE_URL}. Make sure the FastAPI server is running."}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error: {e.response.status_code} - {e.response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request error: {str(e)}"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from API"}

@mcp.tool()
def get_financial_summary(
    account_id: Optional[str] = None,
    analysis_type: str = "comprehensive",
    date_range_days: Optional[int] = None,
    include_predictions: bool = True
) -> str:
    """
    Get an overall summary of all financial accounts stored in the vector database with ML-powered analysis.
    
    Args:
        account_id: Specific account ID to analyze (optional)
        analysis_type: Type of analysis - comprehensive, spending, income, trends (default: comprehensive)
        date_range_days: Number of days to look back for analysis (optional)
        include_predictions: Include spending predictions and trends (default: True)
    
    Returns:
        Comprehensive financial summary with insights and recommendations
    """
    # Prepare request data matching FinancialSummaryQuery model
    request_data = {
        "analysis_type": analysis_type,
        "include_predictions": include_predictions
    }
    
    if account_id:
        request_data["account_id"] = account_id
    if date_range_days:
        request_data["date_range_days"] = date_range_days
    
    result = make_api_request("/financial-summary", method="POST", data=request_data)
    
    if "error" in result:
        return f"Error retrieving financial summary: {result['error']}"
    
    if "detail" in result:
        return f"No financial data found: {result['detail']}"
    
    # Format the comprehensive response
    summary_lines = [
        f"📊 FINANCIAL SUMMARY ({result['summary_type'].upper()})",
        "=" * 60,
        "",
        f"📈 ANALYSIS PERIOD:",
        f"  • Start Date: {result['analysis_period']['start_date']}",
        f"  • End Date: {result['analysis_period']['end_date']}",
        f"  • Days Analyzed: {result['analysis_period']['days_analyzed']}",
        f"  • Accounts Analyzed: {result['account_count']}",
        "",
        f"💰 FINANCIAL HEALTH:",
        f"  • Total Balance: ${result['financial_health']['total_balance']:,.2f}",
        f"  • Net Worth Change: ${result['financial_health']['net_worth_change']:,.2f}",
        f"  • Expense/Income Ratio: {result['financial_health']['expense_to_income_ratio']:.2f}",
        f"  • Financial Stability: {result['financial_health']['financial_stability'].replace('_', ' ').title()}",
        "",
        f"💸 SPENDING ANALYSIS:",
        f"  • Total Expenses: ${result['spending_analysis']['total_expenses']:,.2f}",
        f"  • Average Expense: ${result['spending_analysis']['average_expense']:,.2f}",
        f"  • Largest Expense: ${result['spending_analysis']['largest_expense']:,.2f}",
        f"  • Expense Transactions: {result['spending_analysis']['expense_count']}",
        "",
        f"💵 INCOME ANALYSIS:",
        f"  • Total Income: ${result['income_analysis']['total_income']:,.2f}",
        f"  • Income Transactions: {result['income_analysis']['income_transactions']}",
        f"  • Average Income/Transaction: ${result['income_analysis']['average_income_per_transaction']:,.2f}",
        ""
    ]
    
    # Add spending categories
    if result['spending_analysis'].get('categories'):
        summary_lines.extend([
            f"🏷️  SPENDING CATEGORIES:",
        ])
        for category, stats in result['spending_analysis']['categories'].items():
            summary_lines.append(
                f"  • {category}: ${stats['total']:,.2f} ({stats['percentage']:.1f}%) - {stats['count']} transactions"
            )
        summary_lines.append("")
    
    # Add trends
    balance_trend = result['trends']['balance_trend']
    trend_emoji = "📈" if balance_trend == "increasing" else "📉" if balance_trend == "decreasing" else "➡️"
    summary_lines.extend([
        f"📊 TRENDS:",
        f"  • Balance Trend: {trend_emoji} {balance_trend.title()}",
        ""
    ])
    
    # Add insights
    if result.get('insights'):
        summary_lines.extend([
            f"💡 AI INSIGHTS:",
        ])
        for insight in result['insights']:
            summary_lines.append(f"  • {insight}")
        summary_lines.append("")
    
    # Add recommendations
    if result.get('recommendations'):
        summary_lines.extend([
            f"🎯 RECOMMENDATIONS:",
        ])
        for recommendation in result['recommendations']:
            summary_lines.append(f"  • {recommendation}")
    
    return "\n".join(summary_lines)

# Add this tool to your existing MCP server (after the existing tools)

@mcp.tool()
def get_all_financial_records(
    include_documents: bool = True,
    include_metadata: bool = True,
    include_original_data: bool = False,
    limit: Optional[int] = None,
    account_id_filter: Optional[str] = None,
    format_type: str = "summary"
) -> str:
    """
    Retrieve all financial records from the vector database with comprehensive filtering and formatting options.
    
    Args:
        include_documents: Include full narrative documents (default: True)
        include_metadata: Include metadata like balances, dates, etc. (default: True)
        include_original_data: Include the original transaction data (default: False)
        limit: Maximum number of records to return (optional)
        account_id_filter: Filter by specific account ID (optional)
        format_type: Format of output - 'summary', 'detailed', 'table', 'json' (default: 'summary')
    
    Returns:
        Formatted list of all financial records with comprehensive statistics
    """
    # Prepare query parameters for GET request
    params = {
        "include_documents": str(include_documents).lower(),
        "include_metadata": str(include_metadata).lower(),
        "include_original_data": str(include_original_data).lower()
    }
    
    if limit:
        params["limit"] = str(limit)
    if account_id_filter:
        params["account_id_filter"] = account_id_filter
    
    # Build query string
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    endpoint = f"/all-records?{query_string}"
    
    result = make_api_request(endpoint, method="GET")
    
    if "error" in result:
        return f"❌ Error retrieving all records: {result['error']}"
    
    if not result.get("records"):
        return "📭 No financial records found in the database."
    
    # Format based on requested format type
    if format_type == "json":
        return json.dumps(result, indent=2)
    
    # Build formatted output
    summary = result.get("summary", {})
    records = result.get("records", [])
    
    output_lines = [
        "📊 ALL FINANCIAL RECORDS",
        "=" * 60,
        "",
        "📈 DATABASE OVERVIEW:",
        f"  • Total Records in DB: {result.get('total_records', 0)}",
        f"  • Records Retrieved: {result.get('records_returned', 0)}",
        f"  • Unique Accounts: {summary.get('unique_accounts', 0)}",
        f"  • Total Transactions: {summary.get('total_transactions', 0):,}",
        f"  • Combined Balance: ${summary.get('total_balance_across_accounts', 0):,.2f}",
        ""
    ]
    
    # Add overall date range if available
    if summary.get('overall_date_range'):
        date_range = summary['overall_date_range']
        output_lines.extend([
            f"📅 OVERALL DATE RANGE:",
            f"  • Earliest Transaction: {date_range.get('earliest', 'N/A')}",
            f"  • Latest Transaction: {date_range.get('latest', 'N/A')}",
            ""
        ])
    
    # Add query parameters used
    query_params = summary.get('query_parameters', {})
    output_lines.extend([
        f"⚙️  QUERY PARAMETERS:",
        f"  • Include Documents: {query_params.get('include_documents', 'N/A')}",
        f"  • Include Metadata: {query_params.get('include_metadata', 'N/A')}",
        f"  • Include Original Data: {query_params.get('include_original_data', 'N/A')}",
        f"  • Limit Applied: {query_params.get('limit_applied', 'None')}",
        f"  • Account Filter: {query_params.get('account_filter', 'None')}",
        ""
    ])
    
    # Add account summaries
    if summary.get('account_summary'):
        output_lines.extend([
            f"🏦 ACCOUNT SUMMARIES:",
            ""
        ])
        
        for account_id, account_data in summary['account_summary'].items():
            date_range = account_data.get('date_range', {})
            date_range_str = f"{date_range.get('earliest', 'N/A')} to {date_range.get('latest', 'N/A')}"
            
            output_lines.extend([
                f"  📋 Account: {account_id}",
                f"    • Transaction Count: {account_data.get('transaction_count', 0):,}",
                f"    • Total Balance: ${account_data.get('total_balance', 0):,.2f}",
                f"    • Date Range: {date_range_str}",
                ""
            ])
    
    # Add individual records based on format type
    if format_type in ["detailed", "summary"]:
        output_lines.extend([
            f"📄 INDIVIDUAL RECORDS:",
            ""
        ])
        
        for i, record in enumerate(records, 1):
            doc_id = record.get("document_id", "Unknown")
            relevance = record.get("relevance_score", 0)
            
            output_lines.append(f"  📋 Record {i}: {doc_id}")
            output_lines.append(f"    • Relevance Score: {relevance:.3f}")
            
            # Add metadata if available
            if record.get("metadata"):
                metadata = record["metadata"]
                output_lines.extend([
                    f"    • Account ID: {metadata.get('account_id', 'N/A')}",
                    f"    • Initial Balance: ${metadata.get('initial_balance', 0):,.2f}",
                    f"    • Final Balance: ${metadata.get('final_balance', 0):,.2f}",
                    f"    • Transaction Count: {metadata.get('transaction_count', 0)}",
                    f"    • Total Spent: ${metadata.get('total_spent', 0):,.2f}",
                    f"    • Total Received: ${metadata.get('total_received', 0):,.2f}",
                    f"    • Transaction Types: {metadata.get('transaction_types', 'N/A')}",
                ])

@mcp.tool()
def search_financial_data(
    query: str,
    max_results: int = 5,
    date_filter: Optional[str] = None,
    amount_filter_min: Optional[float] = None,
    amount_filter_max: Optional[float] = None
) -> str:
    """
    Search financial transaction data using natural language queries.
    
    Args:
        query: Natural language query about transactions (e.g., "Show me all grocery expenses", "What were my highest expenses?")
        max_results: Maximum number of results to return (default: 5)
        date_filter: Filter by date in YYYY-MM-DD format (optional)
        amount_filter_min: Minimum amount filter (optional)
        amount_filter_max: Maximum amount filter (optional)
    
    Returns:
        Formatted search results with financial data
    """
    # Prepare request data matching FinancialSearchQuery model
    search_data = {
        "query": query,
        "n_results": max_results
    }
    
    if date_filter:
        search_data["date_filter"] = date_filter
    
    if amount_filter_min is not None or amount_filter_max is not None:
        amount_filter = {}
        if amount_filter_min is not None:
            amount_filter["min"] = amount_filter_min
        if amount_filter_max is not None:
            amount_filter["max"] = amount_filter_max
        search_data["amount_filter"] = amount_filter
    
    result = make_api_request("/search-financial", method="POST", data=search_data)
    
    if "error" in result:
        return f"Error searching financial data: {result['error']}"
    
    if not result.get("results"):
        return f"No financial data found matching query: '{query}'"
    
    # Format the results
    output_lines = [f"🔍 Search Results for: '{query}'"]
    output_lines.append(f"Found {len(result['results'])} matching account(s)")
    output_lines.append("")
    
    for i, account_result in enumerate(result["results"], 1):
        summary = account_result["summary"]
        relevance = account_result["relevance_score"]
        
        output_lines.append(f"📋 Account {i} (Relevance: {relevance:.2f}):")
        date_range_data = json.loads(summary.get('date_range', '{}')) if summary.get('date_range') else {}
        date_range_str = f"{date_range_data.get('earliest', 'N/A')} to {date_range_data.get('latest', 'N/A')}"
        
        output_lines.append(f"  • Initial Balance: ${summary.get('initial_balance', 0):,.2f}")
        output_lines.append(f"  • Final Balance: ${summary.get('final_balance', 0):,.2f}")
        output_lines.append(f"  • Transaction Count: {summary.get('transaction_count', 0)}")
        output_lines.append(f"  • Date Range: {date_range_str}")
        
        # Show some transaction details
        financial_data = account_result.get("financial_data", {})
        transactions = financial_data.get("transactions", [])
        
        if transactions:
            output_lines.append("  • Recent Transactions:")
            for tx in transactions[:3]:  # Show first 3 transactions
                amount_type = "💸 expense" if tx["amount"] < 0 else "💰 income"
                output_lines.append(f"    - {tx['date']}: {tx['description']} (${abs(tx['amount']):,.2f} {amount_type})")
            
            if len(transactions) > 3:
                output_lines.append(f"    ... and {len(transactions) - 3} more transactions")
        
        output_lines.append("")
    
    # Add summary statistics
    summary = result.get("summary", {})
    output_lines.append("📊 Search Summary:")
    output_lines.append(f"  • Total Accounts Found: {summary.get('total_accounts_found', 0)}")
    output_lines.append(f"  • Combined Balance: ${summary.get('combined_balance', 0):,.2f}")
    output_lines.append(f"  • Total Transactions: {summary.get('total_transactions', 0)}")
    
    # Add special handling for transfer expenses if present
    if summary.get('total_transfer_expense'):
        output_lines.append(f"  • Total Transfer Expenses: ${summary.get('total_transfer_expense', 0):,.2f}")
    
    return "\n".join(output_lines)

@mcp.tool()
def store_financial_data(
    initial_balance: float,
    transactions: str,
    account_id: Optional[str] = None,
    metadata: Optional[str] = None
) -> str:
    """
    Store new financial transaction data in the vector database.
    
    Args:
        initial_balance: Starting account balance
        transactions: JSON string of transactions list, each with: date (YYYY-MM-DD), description, type, amount
        account_id: Optional account identifier
        metadata: Optional JSON string with additional metadata
    
    Returns:
        Success message with document ID and summary
    
    Example transactions format:
    '[{"date": "2024-01-15", "description": "Grocery Store", "type": "debit", "amount": -85.50}, {"date": "2024-01-16", "description": "Salary", "type": "credit", "amount": 3000.00}]'
    """
    try:
        # Parse transactions JSON
        transactions_data = json.loads(transactions)
        
        # Prepare the data matching FinancialData model
        financial_data = {
            "initial_balance": initial_balance,
            "transactions": transactions_data
        }
        
        if account_id:
            financial_data["account_id"] = account_id
        
        if metadata:
            financial_data["metadata"] = json.loads(metadata)
        
        result = make_api_request("/store-financial-data", method="POST", data=financial_data)
        
        if "error" in result:
            return f"Error storing financial data: {result['error']}"
        
        summary = result.get("summary", {})
        date_range_data = json.loads(summary.get('date_range', '{}')) if summary.get('date_range') else {}
        date_range_str = f"{date_range_data.get('earliest', 'N/A')} to {date_range_data.get('latest', 'N/A')}"
        
        return f"""✅ Financial data stored successfully!

📋 Storage Details:
  • Document ID: {result.get('document_id')}
  • Transaction Count: {summary.get('transaction_count', 0)}
  • Final Balance: ${summary.get('final_balance', 0):,.2f}
  • Date Range: {date_range_str}
  
🎯 Data is now available for RAG-powered queries and analysis!"""
        
    except json.JSONDecodeError as e:
        return f"❌ Error parsing JSON data: {str(e)}. Please ensure transactions and metadata are valid JSON strings."
    except Exception as e:
        return f"❌ Error: {str(e)}"

@mcp.tool()
def check_financial_db_health() -> str:
    """
    Check the health status of the Financial Vector Database API.
    Returns information about the database connection, document count, and configuration.
    """
    result = make_api_request("/health")
    
    if "error" in result:
        return f"❌ Error checking database health: {result['error']}"
    
    if result.get("status") == "healthy":
        return f"""✅ Financial Database Status: HEALTHY

🔧 System Information:
  • Total Documents: {result.get('total_documents', 0)}
  • Embedding Model: {result.get('embedding_model', 'Unknown')}
  • Database Type: {result.get('database_type', 'Unknown')}
  • Optimized For: {result.get('optimized_for', 'Unknown')}
  
🚀 All systems operational and ready for financial analysis!"""
    else:
        return f"❌ Financial Database Status: UNHEALTHY\nError: {result.get('error', 'Unknown error')}"

@mcp.tool()
def get_financial_insights(
    query: str,
    analysis_type: str = "comprehensive",
    date_range_days: Optional[int] = None
) -> str:
    """
    Get intelligent financial insights by combining search results with ML-powered analysis.
    This tool searches for relevant financial data and provides contextual insights with AI recommendations.
    
    Args:
        query: Question or topic for financial analysis (e.g., "What are my spending patterns?", "How much did I spend on food?")
        analysis_type: Type of analysis to perform (comprehensive, spending, income, trends)
        date_range_days: Number of days to look back for analysis (optional)
    
    Returns:
        Detailed financial insights and analysis with AI-powered recommendations
    """
    # First, search for relevant data
    search_result = search_financial_data(query, max_results=10)
    
    if "Error" in search_result or "No financial data found" in search_result:
        return search_result
    
    # Get comprehensive ML-powered summary for context
    summary_result = get_financial_summary(
        analysis_type=analysis_type,
        date_range_days=date_range_days
    )
    
    insights = [
        f"🧠 AI-POWERED FINANCIAL INSIGHTS",
        f"Query: '{query}'",
        "=" * 60,
        "",
        "🔍 RELEVANT SEARCH RESULTS:",
        search_result,
        "",
        "📊 COMPREHENSIVE ANALYSIS:",
        summary_result,
        "",
        "🎯 CONTEXTUAL ANALYSIS:",
        "This analysis combines vector search results with machine learning insights to provide:",
        "  • Pattern recognition across your transaction history",
        "  • Spending categorization using semantic similarity",
        "  • Trend analysis with predictive elements",
        "  • Personalized recommendations based on your financial behavior",
        "",
        "💡 How to use these insights:",
        "  • Review the spending categories to identify optimization opportunities",
        "  • Monitor the balance trends to understand your financial trajectory",
        "  • Follow the AI recommendations to improve your financial health",
        "  • Use the search results to drill down into specific transaction patterns"
    ]
    
    return "\n".join(insights)

def get_all_transactions() -> str:
    """
    Retrieve all available transactions from the financial vector database.
    Returns a JSON string of all transactions across all accounts.
    """
    # Use the all-records endpoint to get all original data
    params = {
        "include_documents": "false",
        "include_metadata": "false",
        "include_original_data": "true"
    }
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    endpoint = f"/all-records?{query_string}"
    result = make_api_request(endpoint, method="GET")

    if "error" in result:
        return f"❌ Error retrieving transactions: {result['error']}"
    if not result.get("records"):
        return "📭 No transactions found in the database."

    # Collect all transactions from all records
    all_transactions = []
    for record in result["records"]:
        original_data = record.get("original_data")
        if original_data and "transactions" in original_data:
            all_transactions.extend(original_data["transactions"])

    return json.dumps(all_transactions, indent=2)

if __name__ == "__main__":
    mcp.run(transport="stdio")
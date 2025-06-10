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
def get_financial_summary() -> str:
    """
    Get an overall summary of all financial accounts stored in the vector database.
    Returns total accounts, combined balance, total transactions, and other aggregate statistics.
    """
    result = make_api_request("/financial-summary")
    
    if "error" in result:
        return f"Error retrieving financial summary: {result['error']}"
    
    if "message" in result and result["message"] == "No financial data found":
        return "No financial data found in the database."
    
    summary_text = f"""Financial Database Summary:
- Total Accounts: {result.get('total_accounts', 0)}
- Combined Balance: ${result.get('combined_balance', 0):.2f}
- Total Transactions: {result.get('total_transactions', 0)}
- Average Balance per Account: ${result.get('average_balance_per_account', 0):.2f}
- Transaction Types: {', '.join(result.get('transaction_types', []))}"""
    
    return summary_text

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
    
    print(f"Search request: {search_data}")
    print(f"Search response: {result}")
     
    if "error" in result:
        return f"Error searching financial data: {result['error']}"
    
    if not result.get("results"):
        return f"No financial data found matching query: '{query}'"
    
    # Format the results
    output_lines = [f"Search Results for: '{query}'"]
    output_lines.append(f"Found {len(result['results'])} matching account(s)")
    output_lines.append("")
    
    for i, account_result in enumerate(result["results"], 1):
        summary = account_result["summary"]
        relevance = account_result["relevance_score"]
        
        output_lines.append(f"Account {i} (Relevance: {relevance:.2f}):")
        output_lines.append(f"  - Initial Balance: ${summary.get('initial_balance', 0):.2f}")
        output_lines.append(f"  - Final Balance: ${summary.get('final_balance', 0):.2f}")
        output_lines.append(f"  - Transaction Count: {summary.get('transaction_count', 0)}")
        output_lines.append(f"  - Date Range: {summary.get('date_range', 'N/A')}")
        
        # Show some transaction details
        financial_data = account_result.get("financial_data", {})
        transactions = financial_data.get("transactions", [])
        
        if transactions:
            output_lines.append("  - Recent Transactions:")
            for tx in transactions[:3]:  # Show first 3 transactions
                amount_type = "expense" if tx["amount"] < 0 else "income"
                output_lines.append(f"    • {tx['date']}: {tx['description']} (${abs(tx['amount']):.2f} {amount_type})")
            
            if len(transactions) > 3:
                output_lines.append(f"    ... and {len(transactions) - 3} more transactions")
        
        output_lines.append("")
    
    # Add summary statistics
    summary = result.get("summary", {})
    output_lines.append("Overall Search Summary:")
    output_lines.append(f"  - Total Accounts Found: {summary.get('total_accounts_found', 0)}")
    output_lines.append(f"  - Combined Balance: ${summary.get('combined_balance', 0):.2f}")
    output_lines.append(f"  - Total Transactions: {summary.get('total_transactions', 0)}")
    
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
        
        # Prepare the data
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
        return f"""Financial data stored successfully!
Document ID: {result.get('document_id')}
Summary:
  - Transaction Count: {summary.get('transaction_count', 0)}
  - Final Balance: ${summary.get('final_balance', 0):.2f}
  - Date Range: {summary.get('date_range', 'N/A')}"""
        
    except json.JSONDecodeError as e:
        return f"Error parsing JSON data: {str(e)}. Please ensure transactions and metadata are valid JSON strings."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def check_financial_db_health() -> str:
    """
    Check the health status of the Financial Vector Database API.
    Returns information about the database connection, document count, and configuration.
    """
    result = make_api_request("/health")
    
    if "error" in result:
        return f"Error checking database health: {result['error']}"
    
    if result.get("status") == "healthy":
        return f"""Financial Database Status: HEALTHY ✓
- Total Documents: {result.get('total_documents', 0)}
- Embedding Model: {result.get('embedding_model', 'Unknown')}
- Database Type: {result.get('database_type', 'Unknown')}
- Optimized For: {result.get('optimized_for', 'Unknown')}"""
    else:
        return f"Financial Database Status: UNHEALTHY ✗\nError: {result.get('error', 'Unknown error')}"

@mcp.tool()
def get_financial_insights(query: str) -> str:
    """
    Get intelligent financial insights by combining search results with analysis.
    This tool searches for relevant financial data and provides contextual insights.
    
    Args:
        query: Question or topic for financial analysis (e.g., "What are my spending patterns?", "How much did I spend on food?")
    
    Returns:
        Detailed financial insights and analysis
    """
    # First, search for relevant data
    search_result = search_financial_data(query, max_results=10)
    
    if "Error" in search_result or "No financial data found" in search_result:
        return search_result
    
    # Get overall summary for context
    summary_result = get_financial_summary()
    
    insights = [
        f"Financial Insights for: '{query}'",
        "=" * 50,
        "",
        "SEARCH RESULTS:",
        search_result,
        "",
        "OVERALL CONTEXT:",
        summary_result,
        "",
        "ANALYSIS NOTES:",
        "- Use this data to identify spending patterns, budget adherence, and financial trends",
        "- Compare account balances and transaction volumes across different periods",
        "- Look for unusual transactions or spending categories that may need attention"
    ]
    
    return "\n".join(insights)

if __name__ == "__main__":
    mcp.run(transport="stdio")
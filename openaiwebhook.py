from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import time
import os
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASSISTANT_ID = os.getenv('ASSISTANT_ID') 
SERPAPI_KEY = os.getenv('SERPAPI_KEY')

# Validate required environment variables
if not all([OPENAI_API_KEY, ASSISTANT_ID, SERPAPI_KEY]):
    logger.error("Missing required environment variables. Check your .env file.")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

def google_search(query, num_results=5):
    """Search Google using SERPAPI"""
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": num_results,
            "no_cache": "false"
        }
        
        logger.info(f"SERPAPI search: {query}")
        response = requests.get("https://serpapi.com/search", params=params, timeout=30)
        
        if response.status_code == 200:
            results = response.json()
            search_summary = []
            
            # Extract organic results
            if 'organic_results' in results:
                for result in results['organic_results'][:num_results]:
                    search_summary.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'link': result.get('link', ''),
                        'position': result.get('position', 0)
                    })
            
            # Check for knowledge graph info (great for businesses)
            if 'knowledge_graph' in results:
                kg = results['knowledge_graph']
                search_summary.insert(0, {
                    'title': f"Knowledge Graph: {kg.get('title', '')}",
                    'snippet': kg.get('description', ''),
                    'link': kg.get('website', ''),
                    'type': 'knowledge_graph'
                })
            
            logger.info(f"Found {len(search_summary)} search results")
            return search_summary
        else:
            logger.error(f"SERPAPI request failed with status {response.status_code}")
            return [{"error": f"SERPAPI request failed with status {response.status_code}"}]
            
    except Exception as e:
        logger.error(f"SERPAPI error: {str(e)}")
        return [{"error": f"SERPAPI error: {str(e)}"}]

def scrape_website(url):
    """Scrape website content"""
    try:
        if not url:
            return "No website provided"
            
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        logger.info(f"Scraping website: {url}")
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            logger.info(f"Scraped {len(text)} characters from website")
            return text[:2000]  # Limit to 2000 characters
        else:
            logger.warning(f"Failed to scrape website: HTTP {response.status_code}")
            return f"Failed to scrape website: HTTP {response.status_code}"
            
    except Exception as e:
        logger.error(f"Website scraping error: {str(e)}")
        return f"Website scraping error: {str(e)}"

def wait_for_run_completion(thread_id, run_id, max_wait=120):
    """Wait for OpenAI Assistant run to complete"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            logger.info(f"Run status: {run.status}")
            
            if run.status == 'completed':
                return True, "completed"
            elif run.status in ['failed', 'cancelled', 'expired']:
                logger.error(f"Run failed with status: {run.status}")
                return False, run.status
            elif run.status == 'requires_action':
                logger.warning("Run requires action (tool calls)")
                return False, "requires_action"
            
            time.sleep(3)  # Check every 3 seconds
            
        except Exception as e:
            logger.error(f"Error checking run status: {str(e)}")
            return False, f"Error checking run status: {str(e)}"
    
    logger.error("Run timed out")
    return False, "timeout"

@app.route('/analyze-business', methods=['POST'])
def analyze_business():
    """Main endpoint for Clay.com webhook"""
    try:
        data = request.json
        
        # Extract data from Clay
        business_name = data.get('business_name', '').strip()
        website = data.get('website', '').strip()
        industry_hint = data.get('industry', '').strip()
        custom_instructions = data.get('custom_instructions', '').strip()
        
        logger.info(f"Starting analysis for: {business_name} - {website}")
        
        if not business_name and not website:
            return jsonify({
                'error': 'Either business_name or website is required'
            }), 400
        
        # Step 1: Gather web research
        search_results = []
        website_content = ""
        
        # Google search for business info
        if business_name:
            search_query = f'"{business_name}" business industry services'
            if website:
                domain = website.replace('https://', '').replace('http://', '').split('/')[0]
                search_query += f' site:{domain}'
            
            search_results = google_search(search_query, num_results=4)
        
        # Scrape website
        if website:
            website_content = scrape_website(website)
        
        # Step 2: Create OpenAI Assistant thread
        thread = client.beta.threads.create()
        logger.info(f"Created thread: {thread.id}")
        
        # Prepare comprehensive context for the assistant
        research_context = f"""
Business Classification Analysis Request:

Business Name: {business_name if business_name else "Not provided"}
Website: {website if website else "Not provided"}
{f"Industry Context: {industry_hint}" if industry_hint else ""}
{f"Special Instructions: {custom_instructions}" if custom_instructions else ""}

=== EXTERNAL RESEARCH DATA ===

Website Content Analysis:
{website_content if website_content else "No website content available"}

Google Search Results:
"""
        
        # Add search results to context
        if search_results:
            for i, result in enumerate(search_results[:4], 1):
                if 'error' not in result:
                    result_type = result.get('type', 'organic')
                    research_context += f"""
Result {i} ({result_type}):
Title: {result.get('title', 'N/A')}
Summary: {result.get('snippet', 'N/A')}
URL: {result.get('link', 'N/A')}
"""
        else:
            research_context += "\nNo search results available"
        
        research_context += """

=== CLASSIFICATION REQUEST ===

Please analyze this business using your NAICS codes knowledge base and provide:

1. **Primary NAICS Code**: The most appropriate 6-digit NAICS code
2. **NAICS Title**: The official title for this code  
3. **Classification Rationale**: Detailed explanation of why this code fits
4. **Key Business Activities**: Specific activities that support this classification
5. **Alternative Codes**: Any other codes that could apply (with brief explanation)
6. **Confidence Level**: High/Medium/Low and reasoning
7. **Data Sources**: Which external research informed your decision

**Instructions**: Cross-reference the external research above with your internal NAICS database. If external research is limited, rely on your knowledge base but note the limitation.
"""
        
        # Add message to thread
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=research_context
        )
        
        # Step 3: Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID,
            instructions="Use your NAICS codes knowledge base to provide accurate business classification. Be thorough and cite sources when possible."
        )
        
        logger.info(f"Created run: {run.id}")
        
        # Step 4: Wait for completion
        success, status = wait_for_run_completion(thread.id, run.id)
        
        if not success:
            return jsonify({
                'error': f'Assistant run failed: {status}',
                'thread_id': thread.id,
                'run_id': run.id,
                'debug_info': {
                    'business_name': business_name,
                    'website': website,
                    'search_results_count': len(search_results),
                    'website_scraped': bool(website_content)
                }
            }), 500
        
        # Step 5: Get response
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        assistant_response = messages.data[0].content[0].text.value
        
        # Extract citations if available
        citations = []
        if messages.data[0].content[0].text.annotations:
            for annotation in messages.data[0].content[0].text.annotations:
                if hasattr(annotation, 'file_citation'):
                    citations.append({
                        'file_id': annotation.file_citation.file_id,
                        'text': annotation.text if hasattr(annotation, 'text') else 'Referenced file'
                    })
        
        logger.info(f"Analysis completed successfully for {business_name}")
        
        return jsonify({
            'success': True,
            'business_name': business_name,
            'website': website,
            'assistant_response': assistant_response,
            'citations': citations,
            'research_summary': {
                'website_scraped': bool(website_content and "error" not in website_content.lower()),
                'search_results_found': len([r for r in search_results if 'error' not in r]),
                'knowledge_graph_found': any(r.get('type') == 'knowledge_graph' for r in search_results),
                'thread_id': thread.id,
                'processing_time_seconds': round(time.time() - time.time(), 2)
            }
        })
        
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'business_name': data.get('business_name', ''),
            'website': data.get('website', '')
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'service': 'clay-openai-webhook',
        'timestamp': time.time(),
        'config_status': {
            'openai_configured': bool(OPENAI_API_KEY),
            'assistant_configured': bool(ASSISTANT_ID),
            'serpapi_configured': bool(SERPAPI_KEY)
        }
    })

@app.route('/test', methods=['POST']) 
def test_endpoint():
    """Test endpoint for debugging"""
    data = request.json
    return jsonify({
        'received_data': data,
        'timestamp': time.time(),
        'server_status': 'operational'
    })

@app.route('/assistants', methods=['GET'])
def list_assistants():
    """List available assistants for debugging"""
    try:
        assistants = client.beta.assistants.list()
        return jsonify({
            'assistants': [
                {
                    'id': asst.id,
                    'name': asst.name,
                    'model': asst.model,
                    'tools': [tool.type for tool in asst.tools]
                }
                for asst in assistants.data
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 Starting Clay-OpenAI Webhook Server")
    print("=" * 50)
    print("Available endpoints:")
    print("- POST /analyze-business (main Clay endpoint)")
    print("- GET  /health (health check)")
    print("- POST /test (testing)")
    print("- GET  /assistants (list your assistants)")
    print("=" * 50)
    print(f"OpenAI API: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing'}")
    print(f"Assistant ID: {'✅ Configured' if ASSISTANT_ID else '❌ Missing'}")
    print(f"SERPAPI Key: {'✅ Configured' if SERPAPI_KEY else '❌ Missing'}")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=6000, debug=True)

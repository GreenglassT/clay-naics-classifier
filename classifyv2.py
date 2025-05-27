from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import time
import os
from openai import OpenAI
from dotenv import load_dotenv
import logging
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASSISTANT_ID = os.getenv('ASSISTANT_ID')
SERPAPI_KEY = os.getenv('SERPAPI_KEY')

# Validate environment variables
if not all([OPENAI_API_KEY, ASSISTANT_ID, SERPAPI_KEY]):
    logger.error("Missing required environment variables")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

def google_search(query, num_results=4):
    """Search Google using SERPAPI"""
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": num_results,
            "no_cache": "false"
        }
        
        response = requests.get("https://serpapi.com/search", params=params, timeout=30)
        
        if response.status_code == 200:
            results = response.json()
            search_summary = []
            
            # Knowledge graph (structured business data)
            if 'knowledge_graph' in results:
                kg = results['knowledge_graph']
                search_summary.append({
                    'title': f"Knowledge Graph: {kg.get('title', '')}",
                    'snippet': kg.get('description', ''),
                    'link': kg.get('website', ''),
                    'type': 'knowledge_graph'
                })
            
            # Organic results
            if 'organic_results' in results:
                for result in results['organic_results'][:num_results]:
                    search_summary.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'link': result.get('link', ''),
                        'type': 'organic'
                    })
            
            return search_summary
        else:
            return [{"error": f"Search failed: {response.status_code}"}]
            
    except Exception as e:
        logger.error(f"SERPAPI error: {e}")
        return [{"error": f"Search error: {str(e)}"}]

def scrape_website(url):
    """Extract website content"""
    try:
        if not url or url == "N/A":
            return "No website provided"
            
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()
            
            # Extract text
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = ' '.join(chunk for chunk in chunks if chunk)
            
            return clean_text[:2500]  # Limit content
        else:
            return f"Website unavailable (HTTP {response.status_code})"
            
    except Exception as e:
        logger.warning(f"Scraping failed for {url}: {e}")
        return f"Website scraping failed: {str(e)}"

def wait_for_completion(thread_id, run_id, max_wait=180):
    """Poll for assistant completion"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            
            if run.status == 'completed':
                return True, "completed"
            elif run.status in ['failed', 'cancelled', 'expired']:
                return False, run.status
            
            time.sleep(3)  # Check every 3 seconds
            
        except Exception as e:
            return False, f"Polling error: {str(e)}"
    
    return False, "timeout"

def parse_classification_response(classification):
    """Extract structured data from AI response"""
    naics_code = "Not determined"
    naics_title = "Not determined"
    confidence_level = "Unknown"
    
    # Extract NAICS code (6-digit number)
    code_match = re.search(r'(\d{6})', classification)
    if code_match:
        naics_code = code_match.group(1)
    
    # Extract confidence level
    if re.search(r'confidence.*high', classification, re.IGNORECASE):
        confidence_level = "High"
    elif re.search(r'confidence.*medium', classification, re.IGNORECASE):
        confidence_level = "Medium"
    elif re.search(r'confidence.*low', classification, re.IGNORECASE):
        confidence_level = "Low"
    
    # Extract NAICS title with improved patterns
    title_patterns = [
        # Pattern 1: "NAICS Title: Title Name"
        r'(?:NAICS Title|Official Title|Title):\s*([^,\n\r\.]+)',
        
        # Pattern 2: "Code - Title Name"
        r'\d{6}\s*-\s*([^,\n\r\.]+)',
        
        # Pattern 3: "Title Name, is the most fitting" or similar
        r'([A-Z][A-Za-z\s&,-]+(?:Services|Contractors|Transportation|Manufacturing|Management|Construction|Retail|Wholesale|Activities|Operations|Arrangement|Distribution|Sales|Providers|Solutions|Systems|Technology|Development|Consulting|Professional|Administrative|Support|Maintenance|Repair|Installation|Production|Processing|Publishing|Broadcasting|Communications|Information|Finance|Insurance|Real Estate|Accommodation|Food|Entertainment|Recreation|Education|Health|Care|Social|Government|Utilities|Mining|Agriculture|Forestry|Fishing|Hunting)),\s*is\s*(?:the\s*most|most|the)',
        
        # Pattern 4: Standard format with quotes
        r'"([^"]+)"',
        
        # Pattern 5: After "classified as" or "classification:"
        r'(?:classified as|classification:)\s*([^,\n\r\.]+)',
    ]
    
    for pattern in title_patterns:
        title_match = re.search(pattern, classification, re.IGNORECASE)
        if title_match:
            raw_title = title_match.group(1).strip()
            
            # Clean up the title - remove common suffixes/prefixes
            cleaned_title = raw_title
            
            # Remove leading articles and common phrases
            cleaned_title = re.sub(r'^(?:the\s+|a\s+|an\s+)', '', cleaned_title, flags=re.IGNORECASE)
            
            # Remove trailing explanatory text
            cleaned_title = re.sub(r',.*$', '', cleaned_title)  # Remove everything after comma
            cleaned_title = re.sub(r'\s+is\s+.*$', '', cleaned_title, flags=re.IGNORECASE)  # Remove "is the most..."
            cleaned_title = re.sub(r'\s+which\s+.*$', '', cleaned_title, flags=re.IGNORECASE)  # Remove "which..."
            
            # Final cleanup
            cleaned_title = cleaned_title.strip()
            
            if len(cleaned_title) > 5 and cleaned_title != "Not determined":  # Basic validation
                naics_title = cleaned_title
                break
    
    return naics_code, naics_title, confidence_level
    
@app.route('/classify-business', methods=['POST'])
def classify_business():
    """Main Clay.com endpoint - returns structured JSON"""
    try:
        data = request.json
        
        # Input validation
        business_name = data.get('business_name', '').strip()
        website = data.get('website', '').strip()
        industry_hint = data.get('industry_hint', '').strip()
        
        if not business_name and not website:
            return jsonify({
                'success': False,
                'status': 'error',
                'error_message': 'business_name or website required',
                'business_name': '',
                'website': '',
                'industry_hint': '',
                'naics_code': None,
                'naics_title': None,
                'confidence_level': None,
                'full_analysis': None,
                'reasoning_summary': None,
                'website_scraped': False,
                'search_results_count': 0,
                'knowledge_graph_found': False,
                'data_sources_used': 0,
                'citations_count': 0,
                'naics_database_used': False,
                'primary_data_source': None,
                'processing_timestamp': time.time(),
                'thread_id': None,
                'assistant_id': ASSISTANT_ID[:12] + "..." if ASSISTANT_ID else None,
                'quality_score': 0
            }), 400
        
        logger.info(f"Processing: {business_name} | {website}")
        
        # Research phase
        search_results = []
        website_content = ""
        
        # Google search
        if business_name:
            query = f'"{business_name}" business services industry'
            if website:
                domain = website.replace('https://', '').replace('http://', '').split('/')[0]
                query += f' site:{domain}'
            search_results = google_search(query)
        
        # Website scraping
        if website:
            website_content = scrape_website(website)
        
        # Build context for OpenAI
        context = f"""
BUSINESS CLASSIFICATION REQUEST

Company: {business_name or 'Unknown'}
Website: {website or 'None provided'}
{f'Industry Context: {industry_hint}' if industry_hint else ''}

=== EXTERNAL RESEARCH ===

Website Content:
{website_content}

Search Results:
"""
        
        # Add search results
        for i, result in enumerate(search_results[:4], 1):
            if 'error' not in result:
                context += f"""
{i}. {result.get('title', 'N/A')} ({result.get('type', 'organic')})
   {result.get('snippet', 'No description')}
   {result.get('link', '')}
"""
        
        context += """

=== ANALYSIS REQUEST ===

Using your NAICS codes database, provide:

1. **Primary NAICS Code**: Most appropriate 6-digit code
2. **Official Title**: Exact NAICS title
3. **Classification Reasoning**: Why this code fits based on business activities
4. **Supporting Evidence**: Specific details from research that support classification
5. **Alternative Codes**: Other possible codes with brief explanations
6. **Confidence Level**: High/Medium/Low with justification

Format your response clearly with the above structure.
"""
        
        # OpenAI Assistant workflow
        thread = client.beta.threads.create()
        
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=context
        )
        
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID,
            instructions="Provide thorough NAICS classification using external research and your knowledge base."
        )
        
        # Wait for completion
        success, status = wait_for_completion(thread.id, run.id)
        
        if not success:
            return jsonify({
                # Status fields
                'success': False,
                'status': 'failed',
                'error_message': f'Classification failed: {status}',
                
                # Input data (for reference)
                'business_name': business_name,
                'website': website,
                'industry_hint': industry_hint,
                
                # Empty classification results
                'naics_code': None,
                'naics_title': None,
                'confidence_level': None,
                'full_analysis': None,
                'reasoning_summary': None,
                
                # Data quality indicators
                'website_scraped': bool(website_content and 'error' not in website_content.lower()),
                'search_results_count': len([r for r in search_results if 'error' not in r]),
                'knowledge_graph_found': any(r.get('type') == 'knowledge_graph' for r in search_results),
                'data_sources_used': 0,
                'citations_count': 0,
                'naics_database_used': False,
                'primary_data_source': None,
                
                # Technical metadata
                'processing_timestamp': time.time(),
                'thread_id': thread.id,
                'assistant_id': ASSISTANT_ID[:12] + "...",
                'quality_score': 0,
                
                # Debug info
                'debug_info': {
                    'run_id': run.id,
                    'failure_reason': status
                }
            }), 500
        
        # Get response
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        classification = messages.data[0].content[0].text.value
        
        # Extract citations
        citations = []
        if messages.data[0].content[0].text.annotations:
            for annotation in messages.data[0].content[0].text.annotations:
                if hasattr(annotation, 'file_citation'):
                    citations.append({
                        'file_id': annotation.file_citation.file_id,
                        'source': 'NAICS Database'
                    })
        
        # Parse the AI response to extract structured data
        naics_code, naics_title, confidence_level = parse_classification_response(classification)
        
        # Calculate quality metrics
        website_successfully_scraped = bool(website_content and 'error' not in website_content.lower())
        search_results_found = len([r for r in search_results if 'error' not in r])
        knowledge_graph_found = any(r.get('type') == 'knowledge_graph' for r in search_results)
        
        # Quality score calculation (0-100)
        quality_score = (
            (50 if naics_code != "Not determined" else 0) +
            (20 if website_successfully_scraped else 0) +
            (20 if search_results_found > 0 else 0) +
            (10 if len(citations) > 0 else 0)
        )
        
        # Clay-friendly structured response
        return jsonify({
            # Status fields
            'success': True,
            'status': 'completed',
            'error_message': None,
            
            # Input data (for reference)
            'business_name': business_name,
            'website': website,
            'industry_hint': industry_hint,
            
            # Primary classification results
            'naics_code': naics_code,
            'naics_title': naics_title,
            'confidence_level': confidence_level,
            
            # Full AI analysis
            'full_analysis': classification,
            'reasoning_summary': classification[:500] + "..." if len(classification) > 500 else classification,
            
            # Data quality indicators
            'website_scraped': website_successfully_scraped,
            'search_results_count': search_results_found,
            'knowledge_graph_found': knowledge_graph_found,
            'data_sources_used': len([
                s for s in [website_content, search_results]
                if s and (isinstance(s, str) and 'error' not in s.lower() or isinstance(s, list) and len(s) > 0)
            ]),
            
            # Citations and references
            'citations_count': len(citations),
            'naics_database_used': len(citations) > 0,
            'primary_data_source': 'Knowledge Graph' if knowledge_graph_found else 'Web Search',
            
            # Technical metadata
            'processing_timestamp': time.time(),
            'thread_id': thread.id,
            'assistant_id': ASSISTANT_ID[:12] + "...",  # Partial ID for debugging
            
            # Quality score (0-100)
            'quality_score': quality_score
        })
        
    except Exception as e:
        logger.error(f"Server error: {e}")
        return jsonify({
            # Status fields
            'success': False,
            'status': 'error',
            'error_message': f'Server error: {str(e)}',
            
            # Input data (for reference)
            'business_name': data.get('business_name', '') if 'data' in locals() else '',
            'website': data.get('website', '') if 'data' in locals() else '',
            'industry_hint': data.get('industry_hint', '') if 'data' in locals() else '',
            
            # Empty classification results
            'naics_code': None,
            'naics_title': None,
            'confidence_level': None,
            'full_analysis': None,
            'reasoning_summary': None,
            
            # Data quality indicators (all false/zero for errors)
            'website_scraped': False,
            'search_results_count': 0,
            'knowledge_graph_found': False,
            'data_sources_used': 0,
            'citations_count': 0,
            'naics_database_used': False,
            'primary_data_source': None,
            
            # Technical metadata
            'processing_timestamp': time.time(),
            'thread_id': None,
            'assistant_id': ASSISTANT_ID[:12] + "..." if ASSISTANT_ID else None,
            'quality_score': 0
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check for monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'clay-naics-classifier',
        'version': '2.0.0',
        'config': {
            'openai': bool(OPENAI_API_KEY),
            'assistant': bool(ASSISTANT_ID),
            'serpapi': bool(SERPAPI_KEY)
        }
    })

@app.route('/test', methods=['POST'])
def test():
    """Test endpoint"""
    return jsonify({
        'message': 'Webhook is working!',
        'received': request.json,
        'timestamp': time.time()
    })

@app.route('/endpoints', methods=['GET'])
def list_endpoints():
    """List all available endpoints"""
    endpoints = []
    for rule in app.url_map.iter_rules():
        endpoints.append({
            'endpoint': rule.rule,
            'methods': list(rule.methods - {'HEAD', 'OPTIONS'})
        })
    return jsonify({'available_endpoints': endpoints})

if __name__ == '__main__':
    print("🚀 Starting Clay NAICS Classifier v2.0")
    print("=" * 50)
    print("Available endpoints:")
    print("- POST /classify-business (main Clay endpoint)")
    print("- GET  /health (health check)")
    print("- POST /test (testing)")
    print("- GET  /endpoints (list all endpoints)")
    print("=" * 50)
    print(f"OpenAI API: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing'}")
    print(f"Assistant ID: {'✅ Configured' if ASSISTANT_ID else '❌ Missing'}")
    print(f"SERPAPI Key: {'✅ Configured' if SERPAPI_KEY else '❌ Missing'}")
    print("=" * 50)
    print("New features:")
    print("✅ Structured JSON responses for Clay")
    print("✅ Auto-parsing of NAICS codes and titles")
    print("✅ Quality scoring (0-100)")
    print("✅ Confidence level extraction")
    print("✅ Consistent error handling")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True), '', cleaned_title)  # Remove everything after comma
            cleaned_title = re.sub(r'\s+is\s+.*

@app.route('/classify-business', methods=['POST'])
def classify_business():
    """Main Clay.com endpoint - returns structured JSON"""
    try:
        data = request.json
        
        # Input validation
        business_name = data.get('business_name', '').strip()
        website = data.get('website', '').strip()
        industry_hint = data.get('industry_hint', '').strip()
        
        if not business_name and not website:
            return jsonify({
                'success': False,
                'status': 'error',
                'error_message': 'business_name or website required',
                'business_name': '',
                'website': '',
                'industry_hint': '',
                'naics_code': None,
                'naics_title': None,
                'confidence_level': None,
                'full_analysis': None,
                'reasoning_summary': None,
                'website_scraped': False,
                'search_results_count': 0,
                'knowledge_graph_found': False,
                'data_sources_used': 0,
                'citations_count': 0,
                'naics_database_used': False,
                'primary_data_source': None,
                'processing_timestamp': time.time(),
                'thread_id': None,
                'assistant_id': ASSISTANT_ID[:12] + "..." if ASSISTANT_ID else None,
                'quality_score': 0
            }), 400
        
        logger.info(f"Processing: {business_name} | {website}")
        
        # Research phase
        search_results = []
        website_content = ""
        
        # Google search
        if business_name:
            query = f'"{business_name}" business services industry'
            if website:
                domain = website.replace('https://', '').replace('http://', '').split('/')[0]
                query += f' site:{domain}'
            search_results = google_search(query)
        
        # Website scraping
        if website:
            website_content = scrape_website(website)
        
        # Build context for OpenAI
        context = f"""
BUSINESS CLASSIFICATION REQUEST

Company: {business_name or 'Unknown'}
Website: {website or 'None provided'}
{f'Industry Context: {industry_hint}' if industry_hint else ''}

=== EXTERNAL RESEARCH ===

Website Content:
{website_content}

Search Results:
"""
        
        # Add search results
        for i, result in enumerate(search_results[:4], 1):
            if 'error' not in result:
                context += f"""
{i}. {result.get('title', 'N/A')} ({result.get('type', 'organic')})
   {result.get('snippet', 'No description')}
   {result.get('link', '')}
"""
        
        context += """

=== ANALYSIS REQUEST ===

Using your NAICS codes database, provide:

1. **Primary NAICS Code**: Most appropriate 6-digit code
2. **Official Title**: Exact NAICS title
3. **Classification Reasoning**: Why this code fits based on business activities
4. **Supporting Evidence**: Specific details from research that support classification
5. **Alternative Codes**: Other possible codes with brief explanations
6. **Confidence Level**: High/Medium/Low with justification

Format your response clearly with the above structure.
"""
        
        # OpenAI Assistant workflow
        thread = client.beta.threads.create()
        
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=context
        )
        
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID,
            instructions="Provide thorough NAICS classification using external research and your knowledge base."
        )
        
        # Wait for completion
        success, status = wait_for_completion(thread.id, run.id)
        
        if not success:
            return jsonify({
                # Status fields
                'success': False,
                'status': 'failed',
                'error_message': f'Classification failed: {status}',
                
                # Input data (for reference)
                'business_name': business_name,
                'website': website,
                'industry_hint': industry_hint,
                
                # Empty classification results
                'naics_code': None,
                'naics_title': None,
                'confidence_level': None,
                'full_analysis': None,
                'reasoning_summary': None,
                
                # Data quality indicators
                'website_scraped': bool(website_content and 'error' not in website_content.lower()),
                'search_results_count': len([r for r in search_results if 'error' not in r]),
                'knowledge_graph_found': any(r.get('type') == 'knowledge_graph' for r in search_results),
                'data_sources_used': 0,
                'citations_count': 0,
                'naics_database_used': False,
                'primary_data_source': None,
                
                # Technical metadata
                'processing_timestamp': time.time(),
                'thread_id': thread.id,
                'assistant_id': ASSISTANT_ID[:12] + "...",
                'quality_score': 0,
                
                # Debug info
                'debug_info': {
                    'run_id': run.id,
                    'failure_reason': status
                }
            }), 500
        
        # Get response
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        classification = messages.data[0].content[0].text.value
        
        # Extract citations
        citations = []
        if messages.data[0].content[0].text.annotations:
            for annotation in messages.data[0].content[0].text.annotations:
                if hasattr(annotation, 'file_citation'):
                    citations.append({
                        'file_id': annotation.file_citation.file_id,
                        'source': 'NAICS Database'
                    })
        
        # Parse the AI response to extract structured data
        naics_code, naics_title, confidence_level = parse_classification_response(classification)
        
        # Calculate quality metrics
        website_successfully_scraped = bool(website_content and 'error' not in website_content.lower())
        search_results_found = len([r for r in search_results if 'error' not in r])
        knowledge_graph_found = any(r.get('type') == 'knowledge_graph' for r in search_results)
        
        # Quality score calculation (0-100)
        quality_score = (
            (50 if naics_code != "Not determined" else 0) +
            (20 if website_successfully_scraped else 0) +
            (20 if search_results_found > 0 else 0) +
            (10 if len(citations) > 0 else 0)
        )
        
        # Clay-friendly structured response
        return jsonify({
            # Status fields
            'success': True,
            'status': 'completed',
            'error_message': None,
            
            # Input data (for reference)
            'business_name': business_name,
            'website': website,
            'industry_hint': industry_hint,
            
            # Primary classification results
            'naics_code': naics_code,
            'naics_title': naics_title,
            'confidence_level': confidence_level,
            
            # Full AI analysis
            'full_analysis': classification,
            'reasoning_summary': classification[:500] + "..." if len(classification) > 500 else classification,
            
            # Data quality indicators
            'website_scraped': website_successfully_scraped,
            'search_results_count': search_results_found,
            'knowledge_graph_found': knowledge_graph_found,
            'data_sources_used': len([
                s for s in [website_content, search_results]
                if s and (isinstance(s, str) and 'error' not in s.lower() or isinstance(s, list) and len(s) > 0)
            ]),
            
            # Citations and references
            'citations_count': len(citations),
            'naics_database_used': len(citations) > 0,
            'primary_data_source': 'Knowledge Graph' if knowledge_graph_found else 'Web Search',
            
            # Technical metadata
            'processing_timestamp': time.time(),
            'thread_id': thread.id,
            'assistant_id': ASSISTANT_ID[:12] + "...",  # Partial ID for debugging
            
            # Quality score (0-100)
            'quality_score': quality_score
        })
        
    except Exception as e:
        logger.error(f"Server error: {e}")
        return jsonify({
            # Status fields
            'success': False,
            'status': 'error',
            'error_message': f'Server error: {str(e)}',
            
            # Input data (for reference)
            'business_name': data.get('business_name', '') if 'data' in locals() else '',
            'website': data.get('website', '') if 'data' in locals() else '',
            'industry_hint': data.get('industry_hint', '') if 'data' in locals() else '',
            
            # Empty classification results
            'naics_code': None,
            'naics_title': None,
            'confidence_level': None,
            'full_analysis': None,
            'reasoning_summary': None,
            
            # Data quality indicators (all false/zero for errors)
            'website_scraped': False,
            'search_results_count': 0,
            'knowledge_graph_found': False,
            'data_sources_used': 0,
            'citations_count': 0,
            'naics_database_used': False,
            'primary_data_source': None,
            
            # Technical metadata
            'processing_timestamp': time.time(),
            'thread_id': None,
            'assistant_id': ASSISTANT_ID[:12] + "..." if ASSISTANT_ID else None,
            'quality_score': 0
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check for monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'clay-naics-classifier',
        'version': '2.0.0',
        'config': {
            'openai': bool(OPENAI_API_KEY),
            'assistant': bool(ASSISTANT_ID),
            'serpapi': bool(SERPAPI_KEY)
        }
    })

@app.route('/test', methods=['POST'])
def test():
    """Test endpoint"""
    return jsonify({
        'message': 'Webhook is working!',
        'received': request.json,
        'timestamp': time.time()
    })

@app.route('/endpoints', methods=['GET'])
def list_endpoints():
    """List all available endpoints"""
    endpoints = []
    for rule in app.url_map.iter_rules():
        endpoints.append({
            'endpoint': rule.rule,
            'methods': list(rule.methods - {'HEAD', 'OPTIONS'})
        })
    return jsonify({'available_endpoints': endpoints})

if __name__ == '__main__':
    print("🚀 Starting Clay NAICS Classifier v2.0")
    print("=" * 50)
    print("Available endpoints:")
    print("- POST /classify-business (main Clay endpoint)")
    print("- GET  /health (health check)")
    print("- POST /test (testing)")
    print("- GET  /endpoints (list all endpoints)")
    print("=" * 50)
    print(f"OpenAI API: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing'}")
    print(f"Assistant ID: {'✅ Configured' if ASSISTANT_ID else '❌ Missing'}")
    print(f"SERPAPI Key: {'✅ Configured' if SERPAPI_KEY else '❌ Missing'}")
    print("=" * 50)
    print("New features:")
    print("✅ Structured JSON responses for Clay")
    print("✅ Auto-parsing of NAICS codes and titles")
    print("✅ Quality scoring (0-100)")
    print("✅ Confidence level extraction")
    print("✅ Consistent error handling")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True), '', cleaned_title, flags=re.IGNORECASE)  # Remove "is the most..."
            cleaned_title = re.sub(r'\s+which\s+.*

@app.route('/classify-business', methods=['POST'])
def classify_business():
    """Main Clay.com endpoint - returns structured JSON"""
    try:
        data = request.json
        
        # Input validation
        business_name = data.get('business_name', '').strip()
        website = data.get('website', '').strip()
        industry_hint = data.get('industry_hint', '').strip()
        
        if not business_name and not website:
            return jsonify({
                'success': False,
                'status': 'error',
                'error_message': 'business_name or website required',
                'business_name': '',
                'website': '',
                'industry_hint': '',
                'naics_code': None,
                'naics_title': None,
                'confidence_level': None,
                'full_analysis': None,
                'reasoning_summary': None,
                'website_scraped': False,
                'search_results_count': 0,
                'knowledge_graph_found': False,
                'data_sources_used': 0,
                'citations_count': 0,
                'naics_database_used': False,
                'primary_data_source': None,
                'processing_timestamp': time.time(),
                'thread_id': None,
                'assistant_id': ASSISTANT_ID[:12] + "..." if ASSISTANT_ID else None,
                'quality_score': 0
            }), 400
        
        logger.info(f"Processing: {business_name} | {website}")
        
        # Research phase
        search_results = []
        website_content = ""
        
        # Google search
        if business_name:
            query = f'"{business_name}" business services industry'
            if website:
                domain = website.replace('https://', '').replace('http://', '').split('/')[0]
                query += f' site:{domain}'
            search_results = google_search(query)
        
        # Website scraping
        if website:
            website_content = scrape_website(website)
        
        # Build context for OpenAI
        context = f"""
BUSINESS CLASSIFICATION REQUEST

Company: {business_name or 'Unknown'}
Website: {website or 'None provided'}
{f'Industry Context: {industry_hint}' if industry_hint else ''}

=== EXTERNAL RESEARCH ===

Website Content:
{website_content}

Search Results:
"""
        
        # Add search results
        for i, result in enumerate(search_results[:4], 1):
            if 'error' not in result:
                context += f"""
{i}. {result.get('title', 'N/A')} ({result.get('type', 'organic')})
   {result.get('snippet', 'No description')}
   {result.get('link', '')}
"""
        
        context += """

=== ANALYSIS REQUEST ===

Using your NAICS codes database, provide:

1. **Primary NAICS Code**: Most appropriate 6-digit code
2. **Official Title**: Exact NAICS title
3. **Classification Reasoning**: Why this code fits based on business activities
4. **Supporting Evidence**: Specific details from research that support classification
5. **Alternative Codes**: Other possible codes with brief explanations
6. **Confidence Level**: High/Medium/Low with justification

Format your response clearly with the above structure.
"""
        
        # OpenAI Assistant workflow
        thread = client.beta.threads.create()
        
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=context
        )
        
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID,
            instructions="Provide thorough NAICS classification using external research and your knowledge base."
        )
        
        # Wait for completion
        success, status = wait_for_completion(thread.id, run.id)
        
        if not success:
            return jsonify({
                # Status fields
                'success': False,
                'status': 'failed',
                'error_message': f'Classification failed: {status}',
                
                # Input data (for reference)
                'business_name': business_name,
                'website': website,
                'industry_hint': industry_hint,
                
                # Empty classification results
                'naics_code': None,
                'naics_title': None,
                'confidence_level': None,
                'full_analysis': None,
                'reasoning_summary': None,
                
                # Data quality indicators
                'website_scraped': bool(website_content and 'error' not in website_content.lower()),
                'search_results_count': len([r for r in search_results if 'error' not in r]),
                'knowledge_graph_found': any(r.get('type') == 'knowledge_graph' for r in search_results),
                'data_sources_used': 0,
                'citations_count': 0,
                'naics_database_used': False,
                'primary_data_source': None,
                
                # Technical metadata
                'processing_timestamp': time.time(),
                'thread_id': thread.id,
                'assistant_id': ASSISTANT_ID[:12] + "...",
                'quality_score': 0,
                
                # Debug info
                'debug_info': {
                    'run_id': run.id,
                    'failure_reason': status
                }
            }), 500
        
        # Get response
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        classification = messages.data[0].content[0].text.value
        
        # Extract citations
        citations = []
        if messages.data[0].content[0].text.annotations:
            for annotation in messages.data[0].content[0].text.annotations:
                if hasattr(annotation, 'file_citation'):
                    citations.append({
                        'file_id': annotation.file_citation.file_id,
                        'source': 'NAICS Database'
                    })
        
        # Parse the AI response to extract structured data
        naics_code, naics_title, confidence_level = parse_classification_response(classification)
        
        # Calculate quality metrics
        website_successfully_scraped = bool(website_content and 'error' not in website_content.lower())
        search_results_found = len([r for r in search_results if 'error' not in r])
        knowledge_graph_found = any(r.get('type') == 'knowledge_graph' for r in search_results)
        
        # Quality score calculation (0-100)
        quality_score = (
            (50 if naics_code != "Not determined" else 0) +
            (20 if website_successfully_scraped else 0) +
            (20 if search_results_found > 0 else 0) +
            (10 if len(citations) > 0 else 0)
        )
        
        # Clay-friendly structured response
        return jsonify({
            # Status fields
            'success': True,
            'status': 'completed',
            'error_message': None,
            
            # Input data (for reference)
            'business_name': business_name,
            'website': website,
            'industry_hint': industry_hint,
            
            # Primary classification results
            'naics_code': naics_code,
            'naics_title': naics_title,
            'confidence_level': confidence_level,
            
            # Full AI analysis
            'full_analysis': classification,
            'reasoning_summary': classification[:500] + "..." if len(classification) > 500 else classification,
            
            # Data quality indicators
            'website_scraped': website_successfully_scraped,
            'search_results_count': search_results_found,
            'knowledge_graph_found': knowledge_graph_found,
            'data_sources_used': len([
                s for s in [website_content, search_results]
                if s and (isinstance(s, str) and 'error' not in s.lower() or isinstance(s, list) and len(s) > 0)
            ]),
            
            # Citations and references
            'citations_count': len(citations),
            'naics_database_used': len(citations) > 0,
            'primary_data_source': 'Knowledge Graph' if knowledge_graph_found else 'Web Search',
            
            # Technical metadata
            'processing_timestamp': time.time(),
            'thread_id': thread.id,
            'assistant_id': ASSISTANT_ID[:12] + "...",  # Partial ID for debugging
            
            # Quality score (0-100)
            'quality_score': quality_score
        })
        
    except Exception as e:
        logger.error(f"Server error: {e}")
        return jsonify({
            # Status fields
            'success': False,
            'status': 'error',
            'error_message': f'Server error: {str(e)}',
            
            # Input data (for reference)
            'business_name': data.get('business_name', '') if 'data' in locals() else '',
            'website': data.get('website', '') if 'data' in locals() else '',
            'industry_hint': data.get('industry_hint', '') if 'data' in locals() else '',
            
            # Empty classification results
            'naics_code': None,
            'naics_title': None,
            'confidence_level': None,
            'full_analysis': None,
            'reasoning_summary': None,
            
            # Data quality indicators (all false/zero for errors)
            'website_scraped': False,
            'search_results_count': 0,
            'knowledge_graph_found': False,
            'data_sources_used': 0,
            'citations_count': 0,
            'naics_database_used': False,
            'primary_data_source': None,
            
            # Technical metadata
            'processing_timestamp': time.time(),
            'thread_id': None,
            'assistant_id': ASSISTANT_ID[:12] + "..." if ASSISTANT_ID else None,
            'quality_score': 0
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check for monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'clay-naics-classifier',
        'version': '2.0.0',
        'config': {
            'openai': bool(OPENAI_API_KEY),
            'assistant': bool(ASSISTANT_ID),
            'serpapi': bool(SERPAPI_KEY)
        }
    })

@app.route('/test', methods=['POST'])
def test():
    """Test endpoint"""
    return jsonify({
        'message': 'Webhook is working!',
        'received': request.json,
        'timestamp': time.time()
    })

@app.route('/endpoints', methods=['GET'])
def list_endpoints():
    """List all available endpoints"""
    endpoints = []
    for rule in app.url_map.iter_rules():
        endpoints.append({
            'endpoint': rule.rule,
            'methods': list(rule.methods - {'HEAD', 'OPTIONS'})
        })
    return jsonify({'available_endpoints': endpoints})

if __name__ == '__main__':
    print("🚀 Starting Clay NAICS Classifier v2.0")
    print("=" * 50)
    print("Available endpoints:")
    print("- POST /classify-business (main Clay endpoint)")
    print("- GET  /health (health check)")
    print("- POST /test (testing)")
    print("- GET  /endpoints (list all endpoints)")
    print("=" * 50)
    print(f"OpenAI API: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing'}")
    print(f"Assistant ID: {'✅ Configured' if ASSISTANT_ID else '❌ Missing'}")
    print(f"SERPAPI Key: {'✅ Configured' if SERPAPI_KEY else '❌ Missing'}")
    print("=" * 50)
    print("New features:")
    print("✅ Structured JSON responses for Clay")
    print("✅ Auto-parsing of NAICS codes and titles")
    print("✅ Quality scoring (0-100)")
    print("✅ Confidence level extraction")
    print("✅ Consistent error handling")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True), '', cleaned_title, flags=re.IGNORECASE)  # Remove "which..."
            
            # Final cleanup
            cleaned_title = cleaned_title.strip()
            
            if len(cleaned_title) > 5 and cleaned_title != "Not determined":  # Basic validation
                naics_title = cleaned_title
                break
    
    return naics_code, naics_title, confidence_level

@app.route('/classify-business', methods=['POST'])
def classify_business():
    """Main Clay.com endpoint - returns structured JSON"""
    try:
        data = request.json
        
        # Input validation
        business_name = data.get('business_name', '').strip()
        website = data.get('website', '').strip()
        industry_hint = data.get('industry_hint', '').strip()
        
        if not business_name and not website:
            return jsonify({
                'success': False,
                'status': 'error',
                'error_message': 'business_name or website required',
                'business_name': '',
                'website': '',
                'industry_hint': '',
                'naics_code': None,
                'naics_title': None,
                'confidence_level': None,
                'full_analysis': None,
                'reasoning_summary': None,
                'website_scraped': False,
                'search_results_count': 0,
                'knowledge_graph_found': False,
                'data_sources_used': 0,
                'citations_count': 0,
                'naics_database_used': False,
                'primary_data_source': None,
                'processing_timestamp': time.time(),
                'thread_id': None,
                'assistant_id': ASSISTANT_ID[:12] + "..." if ASSISTANT_ID else None,
                'quality_score': 0
            }), 400
        
        logger.info(f"Processing: {business_name} | {website}")
        
        # Research phase
        search_results = []
        website_content = ""
        
        # Google search
        if business_name:
            query = f'"{business_name}" business services industry'
            if website:
                domain = website.replace('https://', '').replace('http://', '').split('/')[0]
                query += f' site:{domain}'
            search_results = google_search(query)
        
        # Website scraping
        if website:
            website_content = scrape_website(website)
        
        # Build context for OpenAI
        context = f"""
BUSINESS CLASSIFICATION REQUEST

Company: {business_name or 'Unknown'}
Website: {website or 'None provided'}
{f'Industry Context: {industry_hint}' if industry_hint else ''}

=== EXTERNAL RESEARCH ===

Website Content:
{website_content}

Search Results:
"""
        
        # Add search results
        for i, result in enumerate(search_results[:4], 1):
            if 'error' not in result:
                context += f"""
{i}. {result.get('title', 'N/A')} ({result.get('type', 'organic')})
   {result.get('snippet', 'No description')}
   {result.get('link', '')}
"""
        
        context += """

=== ANALYSIS REQUEST ===

Using your NAICS codes database, provide:

1. **Primary NAICS Code**: Most appropriate 6-digit code
2. **Official Title**: Exact NAICS title
3. **Classification Reasoning**: Why this code fits based on business activities
4. **Supporting Evidence**: Specific details from research that support classification
5. **Alternative Codes**: Other possible codes with brief explanations
6. **Confidence Level**: High/Medium/Low with justification

Format your response clearly with the above structure.
"""
        
        # OpenAI Assistant workflow
        thread = client.beta.threads.create()
        
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=context
        )
        
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID,
            instructions="Provide thorough NAICS classification using external research and your knowledge base."
        )
        
        # Wait for completion
        success, status = wait_for_completion(thread.id, run.id)
        
        if not success:
            return jsonify({
                # Status fields
                'success': False,
                'status': 'failed',
                'error_message': f'Classification failed: {status}',
                
                # Input data (for reference)
                'business_name': business_name,
                'website': website,
                'industry_hint': industry_hint,
                
                # Empty classification results
                'naics_code': None,
                'naics_title': None,
                'confidence_level': None,
                'full_analysis': None,
                'reasoning_summary': None,
                
                # Data quality indicators
                'website_scraped': bool(website_content and 'error' not in website_content.lower()),
                'search_results_count': len([r for r in search_results if 'error' not in r]),
                'knowledge_graph_found': any(r.get('type') == 'knowledge_graph' for r in search_results),
                'data_sources_used': 0,
                'citations_count': 0,
                'naics_database_used': False,
                'primary_data_source': None,
                
                # Technical metadata
                'processing_timestamp': time.time(),
                'thread_id': thread.id,
                'assistant_id': ASSISTANT_ID[:12] + "...",
                'quality_score': 0,
                
                # Debug info
                'debug_info': {
                    'run_id': run.id,
                    'failure_reason': status
                }
            }), 500
        
        # Get response
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        classification = messages.data[0].content[0].text.value
        
        # Extract citations
        citations = []
        if messages.data[0].content[0].text.annotations:
            for annotation in messages.data[0].content[0].text.annotations:
                if hasattr(annotation, 'file_citation'):
                    citations.append({
                        'file_id': annotation.file_citation.file_id,
                        'source': 'NAICS Database'
                    })
        
        # Parse the AI response to extract structured data
        naics_code, naics_title, confidence_level = parse_classification_response(classification)
        
        # Calculate quality metrics
        website_successfully_scraped = bool(website_content and 'error' not in website_content.lower())
        search_results_found = len([r for r in search_results if 'error' not in r])
        knowledge_graph_found = any(r.get('type') == 'knowledge_graph' for r in search_results)
        
        # Quality score calculation (0-100)
        quality_score = (
            (50 if naics_code != "Not determined" else 0) +
            (20 if website_successfully_scraped else 0) +
            (20 if search_results_found > 0 else 0) +
            (10 if len(citations) > 0 else 0)
        )
        
        # Clay-friendly structured response
        return jsonify({
            # Status fields
            'success': True,
            'status': 'completed',
            'error_message': None,
            
            # Input data (for reference)
            'business_name': business_name,
            'website': website,
            'industry_hint': industry_hint,
            
            # Primary classification results
            'naics_code': naics_code,
            'naics_title': naics_title,
            'confidence_level': confidence_level,
            
            # Full AI analysis
            'full_analysis': classification,
            'reasoning_summary': classification[:500] + "..." if len(classification) > 500 else classification,
            
            # Data quality indicators
            'website_scraped': website_successfully_scraped,
            'search_results_count': search_results_found,
            'knowledge_graph_found': knowledge_graph_found,
            'data_sources_used': len([
                s for s in [website_content, search_results]
                if s and (isinstance(s, str) and 'error' not in s.lower() or isinstance(s, list) and len(s) > 0)
            ]),
            
            # Citations and references
            'citations_count': len(citations),
            'naics_database_used': len(citations) > 0,
            'primary_data_source': 'Knowledge Graph' if knowledge_graph_found else 'Web Search',
            
            # Technical metadata
            'processing_timestamp': time.time(),
            'thread_id': thread.id,
            'assistant_id': ASSISTANT_ID[:12] + "...",  # Partial ID for debugging
            
            # Quality score (0-100)
            'quality_score': quality_score
        })
        
    except Exception as e:
        logger.error(f"Server error: {e}")
        return jsonify({
            # Status fields
            'success': False,
            'status': 'error',
            'error_message': f'Server error: {str(e)}',
            
            # Input data (for reference)
            'business_name': data.get('business_name', '') if 'data' in locals() else '',
            'website': data.get('website', '') if 'data' in locals() else '',
            'industry_hint': data.get('industry_hint', '') if 'data' in locals() else '',
            
            # Empty classification results
            'naics_code': None,
            'naics_title': None,
            'confidence_level': None,
            'full_analysis': None,
            'reasoning_summary': None,
            
            # Data quality indicators (all false/zero for errors)
            'website_scraped': False,
            'search_results_count': 0,
            'knowledge_graph_found': False,
            'data_sources_used': 0,
            'citations_count': 0,
            'naics_database_used': False,
            'primary_data_source': None,
            
            # Technical metadata
            'processing_timestamp': time.time(),
            'thread_id': None,
            'assistant_id': ASSISTANT_ID[:12] + "..." if ASSISTANT_ID else None,
            'quality_score': 0
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check for monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'clay-naics-classifier',
        'version': '2.0.0',
        'config': {
            'openai': bool(OPENAI_API_KEY),
            'assistant': bool(ASSISTANT_ID),
            'serpapi': bool(SERPAPI_KEY)
        }
    })

@app.route('/test', methods=['POST'])
def test():
    """Test endpoint"""
    return jsonify({
        'message': 'Webhook is working!',
        'received': request.json,
        'timestamp': time.time()
    })

@app.route('/endpoints', methods=['GET'])
def list_endpoints():
    """List all available endpoints"""
    endpoints = []
    for rule in app.url_map.iter_rules():
        endpoints.append({
            'endpoint': rule.rule,
            'methods': list(rule.methods - {'HEAD', 'OPTIONS'})
        })
    return jsonify({'available_endpoints': endpoints})

if __name__ == '__main__':
    print("🚀 Starting Clay NAICS Classifier v2.0")
    print("=" * 50)
    print("Available endpoints:")
    print("- POST /classify-business (main Clay endpoint)")
    print("- GET  /health (health check)")
    print("- POST /test (testing)")
    print("- GET  /endpoints (list all endpoints)")
    print("=" * 50)
    print(f"OpenAI API: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing'}")
    print(f"Assistant ID: {'✅ Configured' if ASSISTANT_ID else '❌ Missing'}")
    print(f"SERPAPI Key: {'✅ Configured' if SERPAPI_KEY else '❌ Missing'}")
    print("=" * 50)
    print("New features:")
    print("✅ Structured JSON responses for Clay")
    print("✅ Auto-parsing of NAICS codes and titles")
    print("✅ Quality scoring (0-100)")
    print("✅ Confidence level extraction")
    print("✅ Consistent error handling")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=6000, debug=True)

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

# ICP Configuration - Define your target industries
ICP_NAICS_CODES = {
    # Technology & Software
    "541511": {"title": "Custom Computer Programming Services", "priority": "High", "reason": "Heavy technology adoption"},
    "541512": {"title": "Computer Systems Design Services", "priority": "High", "reason": "Technology-focused business model"},
    "518210": {"title": "Data Processing, Hosting, and Related Services", "priority": "High", "reason": "Cloud infrastructure needs"},
    
    # Professional Services
    "541110": {"title": "Offices of Lawyers", "priority": "Medium", "reason": "Regular technology upgrades"},
    "541211": {"title": "Offices of Certified Public Accountants", "priority": "Medium", "reason": "Compliance software needs"},
    "541611": {"title": "Administrative Management and General Management Consulting Services", "priority": "High", "reason": "Digital transformation focus"},
    
    # Healthcare
    "621111": {"title": "Offices of Physicians (except Mental Health)", "priority": "Medium", "reason": "EMR and patient management systems"},
    "621210": {"title": "Offices of Dentists", "priority": "Low", "reason": "Limited tech adoption"},
    
    # Manufacturing (exclude)
    "311": {"title": "Food Manufacturing", "priority": "Exclude", "reason": "Outside our expertise"},
    "312": {"title": "Beverage and Tobacco Product Manufacturing", "priority": "Exclude", "reason": "Regulatory complexity"},
    
    # Construction (exclude most)
    "236": {"title": "Construction of Buildings", "priority": "Exclude", "reason": "Low technology adoption"},
    "238220": {"title": "Plumbing, Heating, and Air-Conditioning Contractors", "priority": "Low", "reason": "Limited software needs"},
}

# ICP Qualification function
def qualify_against_icp(naics_code, naics_title):
    """
    Qualify business against ICP criteria
    Returns qualification status and reasoning
    """
    qualification = {
        'is_icp_match': False,
        'priority_level': 'Unknown',
        'qualification_reason': 'No ICP data available',
        'icp_category': 'Unqualified'
    }
    
    if not naics_code or naics_code == "Not determined":
        qualification.update({
            'qualification_reason': 'NAICS code not determined',
            'icp_category': 'Unqualified'
        })
        return qualification
    
    # Direct code match
    if naics_code in ICP_NAICS_CODES:
        icp_data = ICP_NAICS_CODES[naics_code]
        qualification.update({
            'is_icp_match': True,
            'priority_level': icp_data['priority'],
            'qualification_reason': icp_data['reason'],
            'icp_category': 'Direct Match'
        })
        return qualification
    
    # Check for partial matches (first 3 digits for industry group)
    industry_group = naics_code[:3]
    for code, data in ICP_NAICS_CODES.items():
        if code.startswith(industry_group) or industry_group == code:
            qualification.update({
                'is_icp_match': True,
                'priority_level': data['priority'],
                'qualification_reason': f"Industry group match: {data['reason']}",
                'icp_category': 'Industry Group Match'
            })
            return qualification
    
    # Check for keyword matches in title
    icp_keywords = {
        'High': ['software', 'technology', 'consulting', 'digital', 'cloud', 'saas'],
        'Medium': ['professional', 'services', 'legal', 'accounting', 'healthcare'],
        'Low': ['retail', 'restaurant', 'construction', 'manufacturing'],
        'Exclude': ['agriculture', 'mining', 'utilities', 'government']
    }
    
    if naics_title and naics_title != "Not determined":
        title_lower = naics_title.lower()
        for priority, keywords in icp_keywords.items():
            for keyword in keywords:
                if keyword in title_lower:
                    qualification.update({
                        'is_icp_match': priority != 'Exclude',
                        'priority_level': priority,
                        'qualification_reason': f"Keyword match: '{keyword}' in business title",
                        'icp_category': 'Keyword Match'
                    })
                    return qualification
    
    # Default: Not in ICP
    qualification.update({
        'is_icp_match': False,
        'priority_level': 'Not Qualified',
        'qualification_reason': 'Business does not match ICP criteria',
        'icp_category': 'Outside ICP'
    })
    
    return qualification

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
    
    # Extract NAICS title - simple approach to avoid syntax issues
    title_patterns = [
        # Pattern 1: "NAICS Title: Title Name"
        r'(?:NAICS Title|Official Title|Title):\s*([^,\n\r\.]+)',
        
        # Pattern 2: "Code - Title Name"
        r'\d{6}\s*-\s*([^,\n\r\.]+)',
        
        # Pattern 3: Look for titles before commas
        r'([A-Z][A-Za-z\s&,-]+(?:Services|Contractors|Transportation|Manufacturing|Management|Construction|Retail|Wholesale|Activities|Operations|Arrangement|Distribution|Sales|Providers|Solutions|Systems|Technology|Development|Consulting|Professional|Administrative|Support|Maintenance|Repair|Installation|Production|Processing|Publishing|Broadcasting|Communications|Information|Finance|Insurance|Real Estate|Accommodation|Food|Entertainment|Recreation|Education|Health|Care|Social|Government|Utilities|Mining|Agriculture|Forestry|Fishing|Hunting)),',
        
        # Pattern 4: Quoted titles
        r'"([^"]+)"',
    ]
    
    for pattern in title_patterns:
        title_match = re.search(pattern, classification, re.IGNORECASE)
        if title_match:
            raw_title = title_match.group(1).strip()
            
            # Clean up the title
            cleaned_title = raw_title
            
            # Remove common prefixes
            if cleaned_title.lower().startswith(('the ', 'a ', 'an ')):
                cleaned_title = ' '.join(cleaned_title.split()[1:])
            
            # Remove everything after comma
            if ',' in cleaned_title:
                cleaned_title = cleaned_title.split(',')[0]
            
            # Final cleanup
            cleaned_title = cleaned_title.strip()
            
            if len(cleaned_title) > 5 and cleaned_title != "Not determined":
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
        
        # NEW: ICP Qualification Step
        icp_qualification = qualify_against_icp(naics_code, naics_title)
        
        # Calculate quality metrics
        website_successfully_scraped = bool(website_content and 'error' not in website_content.lower())
        search_results_found = len([r for r in search_results if 'error' not in r])
        knowledge_graph_found = any(r.get('type') == 'knowledge_graph' for r in search_results)
        
        # Enhanced quality score that includes ICP match
        base_quality_score = (
            (50 if naics_code != "Not determined" else 0) +
            (20 if website_successfully_scraped else 0) +
            (20 if search_results_found > 0 else 0) +
            (10 if len(citations) > 0 else 0)
        )
        
        # ICP bonus scoring
        icp_bonus = 0
        if icp_qualification['is_icp_match']:
            priority_bonuses = {'High': 15, 'Medium': 10, 'Low': 5}
            icp_bonus = priority_bonuses.get(icp_qualification['priority_level'], 0)
        
        quality_score = min(100, base_quality_score + icp_bonus)
        
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
            
            # NEW: ICP Qualification Results
            'is_icp_match': icp_qualification['is_icp_match'],
            'icp_priority': icp_qualification['priority_level'],
            'icp_reason': icp_qualification['qualification_reason'],
            'icp_category': icp_qualification['icp_category'],
            'lead_score': 'Hot' if icp_qualification['priority_level'] == 'High' else
                         'Warm' if icp_qualification['priority_level'] == 'Medium' else
                         'Cold' if icp_qualification['priority_level'] == 'Low' else 'Disqualified',
            
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
            
            # Enhanced quality score (includes ICP bonus)
            'quality_score': quality_score,
            'base_quality_score': base_quality_score,
            'icp_bonus_points': icp_bonus
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

@app.route('/icp-config', methods=['GET'])
def get_icp_config():
    """View current ICP configuration"""
    return jsonify({
        'icp_naics_codes': ICP_NAICS_CODES,
        'total_codes': len(ICP_NAICS_CODES),
        'priority_breakdown': {
            'High': len([c for c in ICP_NAICS_CODES.values() if c['priority'] == 'High']),
            'Medium': len([c for c in ICP_NAICS_CODES.values() if c['priority'] == 'Medium']),
            'Low': len([c for c in ICP_NAICS_CODES.values() if c['priority'] == 'Low']),
            'Exclude': len([c for c in ICP_NAICS_CODES.values() if c['priority'] == 'Exclude'])
        }
    })

@app.route('/test-icp', methods=['POST'])
def test_icp_qualification():
    """Test ICP qualification for a specific NAICS code"""
    data = request.json
    naics_code = data.get('naics_code', '')
    naics_title = data.get('naics_title', '')
    
    if not naics_code:
        return jsonify({'error': 'naics_code required'}), 400
    
    qualification = qualify_against_icp(naics_code, naics_title)
    
    return jsonify({
        'input': {'naics_code': naics_code, 'naics_title': naics_title},
        'qualification': qualification,
        'lead_score': 'Hot' if qualification['priority_level'] == 'High' else
                     'Warm' if qualification['priority_level'] == 'Medium' else
                     'Cold' if qualification['priority_level'] == 'Low' else 'Disqualified'
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
    print("🚀 Starting Clay NAICS Classifier v2.1 with ICP Qualification")
    print("=" * 50)
    print("Available endpoints:")
    print("- POST /classify-business (main Clay endpoint)")
    print("- GET  /health (health check)")
    print("- POST /test (testing)")
    print("- GET  /icp-config (view ICP settings)")
    print("- POST /test-icp (test ICP qualification)")
    print("- GET  /endpoints (list all endpoints)")
    print("=" * 50)
    print(f"OpenAI API: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing'}")
    print(f"Assistant ID: {'✅ Configured' if ASSISTANT_ID else '❌ Missing'}")
    print(f"SERPAPI Key: {'✅ Configured' if SERPAPI_KEY else '❌ Missing'}")
    print("=" * 50)
    print("ICP Configuration:")
    priority_counts = {}
    for code_data in ICP_NAICS_CODES.values():
        priority = code_data['priority']
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    
    for priority, count in priority_counts.items():
        print(f"  {priority}: {count} codes")
    print("=" * 50)
    print("New features:")
    print("✅ Structured JSON responses for Clay")
    print("✅ Auto-parsing of NAICS codes and titles")
    print("✅ Quality scoring (0-100)")
    print("✅ Clean NAICS titles (no extra text)")
    print("✅ Confidence level extraction")
    print("✅ ICP qualification and lead scoring")
    print("✅ Priority-based lead qualification")
    print("✅ Consistent error handling")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=6000, debug=True)

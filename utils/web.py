import os
import re
import html
import json
import requests
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse, quote
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import logging
from functools import lru_cache
import time

load_dotenv()
logger = logging.getLogger(__name__)

# API Configuration
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Alternative search APIs (fallbacks)
SEARX_INSTANCES = [
    "https://searx.be",
    "https://searx.info",
    "https://searx.xyz"
]

# Cache configuration
CACHE_DIR = "data/web_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class WebSearcher:
    def __init__(self):
        self.session = None
        self.cache = {}
        self.rate_limits = {
            "brave": {"calls": 0, "reset_time": time.time() + 3600},
            "searx": {"calls": 0, "reset_time": time.time() + 3600}
        }
        self.max_calls_per_hour = {
            "brave": 100,
            "searx": 50
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

def search_web(query: str, count: int = 5, search_type: str = "general") -> List[str]:
    """
    Enhanced web search with multiple providers and caching
    """
    # Check cache first
    cache_key = f"{query}_{count}_{search_type}"
    cached_result = _get_cached_search(cache_key)
    if cached_result:
        logger.debug(f"Returning cached result for: {query}")
        return cached_result
    
    # Try primary search provider
    results = _search_brave(query, count, search_type)
    
    # Fallback to alternative providers if needed
    if not results or results == ["No useful search results found."]:
        logger.warning(f"Brave search failed for: {query}, trying alternatives")
        results = _search_searx(query, count)
    
    # Cache successful results
    if results and results != ["No useful search results found."]:
        _cache_search_result(cache_key, results)
    
    return results

def _search_brave(query: str, count: int, search_type: str) -> List[str]:
    """Search using Brave Search API with enhanced parameters"""
    if not BRAVE_API_KEY:
        logger.error("Brave API key not configured")
        return []
    
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    
    # Customize parameters based on search type
    params = {
        "q": query,
        "count": count,
        "search_lang": "en",
        "ui_lang": "en-US"
    }
    
    if search_type == "news":
        params["freshness"] = "pd"  # Past day
        params["news"] = True
    elif search_type == "academic":
        query += " site:scholar.google.com OR site:arxiv.org OR site:pubmed.ncbi.nlm.nih.gov"
    elif search_type == "code":
        query += " site:github.com OR site:stackoverflow.com"
    
    try:
        response = requests.get(
            BRAVE_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        
        # Process web results
        for entry in data.get("web", {}).get("results", []):
            title = clean_html(entry.get("title", ""))
            desc = clean_html(entry.get("description", ""))
            url = entry.get("url", "")
            
            # Extract additional metadata
            published = entry.get("published_date", "")
            domain = urlparse(url).netloc
            
            snippet = f"{title}: {desc}"
            if published:
                snippet += f" (Published: {published})"
            snippet += f" [{domain}]"
            
            results.append(snippet)
        
        # Include news results if available
        for entry in data.get("news", {}).get("results", []):
            title = clean_html(entry.get("title", ""))
            desc = clean_html(entry.get("description", ""))
            source = entry.get("source", "")
            
            snippet = f"[NEWS] {title}: {desc} (Source: {source})"
            results.append(snippet)
        
        # Include infobox if available
        infobox = data.get("infobox")
        if infobox:
            info_snippet = f"[INFO] {infobox.get('title', '')}: {infobox.get('description', '')}"
            results.insert(0, info_snippet)
        
        return results if results else ["No useful search results found."]
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Brave search error: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in Brave search: {str(e)}")
        return []

def _search_searx(query: str, count: int) -> List[str]:
    """Fallback search using SearX instances"""
    for instance in SEARX_INSTANCES:
        try:
            params = {
                "q": query,
                "format": "json",
                "categories": "general",
                "engines": "google,bing,duckduckgo",
                "pageno": 1
            }
            
            response = requests.get(
                f"{instance}/search",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for entry in data.get("results", [])[:count]:
                    title = entry.get("title", "")
                    content = entry.get("content", "")
                    url = entry.get("url", "")
                    engine = entry.get("engine", "")
                    
                    snippet = f"{title}: {content} [{engine}]"
                    results.append(clean_html(snippet))
                
                if results:
                    return results
                    
        except Exception as e:
            logger.debug(f"SearX instance {instance} failed: {e}")
            continue
    
    return ["Search failed across all providers."]

async def search_web_async(query: str, count: int = 5) -> List[str]:
    """Asynchronous web search for better performance"""
    async with WebSearcher() as searcher:
        return await searcher._search_async(query, count)

def extract_content(url: str, timeout: int = 10) -> Optional[str]:
    """Extract main content from a webpage"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; NovaBot/1.0)"
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find main content
        main_content = None
        
        # Look for common content containers
        content_selectors = [
            "main", "article", '[role="main"]',
            '.content', '#content', '.post', '.entry-content'
        ]
        
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                main_content = element
                break
        
        # Fallback to body if no main content found
        if not main_content:
            main_content = soup.body
        
        if main_content:
            # Extract text with some formatting preserved
            lines = []
            for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                text = element.get_text(strip=True)
                if text:
                    if element.name.startswith('h'):
                        lines.append(f"\n## {text}\n")
                    elif element.name == 'li':
                        lines.append(f"• {text}")
                    else:
                        lines.append(text)
            
            return "\n".join(lines)
        
        # Last resort: get all text
        return soup.get_text(separator="\n", strip=True)
        
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {e}")
        return None

def extract_structured_data(url: str) -> Dict[str, Any]:
    """Extract structured data (JSON-LD, OpenGraph, etc.) from a webpage"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        structured_data = {}
        
        # Extract JSON-LD
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        if json_ld_scripts:
            structured_data['json_ld'] = []
            for script in json_ld_scripts:
                try:
                    data = json.loads(script.string)
                    structured_data['json_ld'].append(data)
                except:
                    pass
        
        # Extract OpenGraph data
        og_data = {}
        for tag in soup.find_all('meta', property=re.compile(r'^og:')):
            og_data[tag.get('property')[3:]] = tag.get('content')
        
        if og_data:
            structured_data['opengraph'] = og_data
        
        # Extract Twitter Card data
        twitter_data = {}
        for tag in soup.find_all('meta', attrs={'name': re.compile(r'^twitter:')}):
            twitter_data[tag.get('name')[8:]] = tag.get('content')
        
        if twitter_data:
            structured_data['twitter_card'] = twitter_data
        
        # Extract basic metadata
        structured_data['metadata'] = {
            'title': soup.title.string if soup.title else '',
            'description': '',
            'keywords': '',
            'author': ''
        }
        
        # Get description
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag:
            structured_data['metadata']['description'] = desc_tag.get('content', '')
        
        # Get keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag:
            structured_data['metadata']['keywords'] = keywords_tag.get('content', '')
        
        # Get author
        author_tag = soup.find('meta', attrs={'name': 'author'})
        if author_tag:
            structured_data['metadata']['author'] = author_tag.get('content', '')
        
        return structured_data
        
    except Exception as e:
        logger.error(f"Error extracting structured data from {url}: {e}")
        return {}

def search_academic(query: str, count: int = 5) -> List[Dict[str, Any]]:
    """Search for academic papers and return structured results"""
    academic_results = []
    
    # Search multiple academic sources
    sources = {
        "arxiv": f"site:arxiv.org {query}",
        "scholar": f"site:scholar.google.com {query}",
        "pubmed": f"site:pubmed.ncbi.nlm.nih.gov {query}"
    }
    
    for source_name, search_query in sources.items():
        results = search_web(search_query, count=count//len(sources) + 1, search_type="academic")
        
        for result in results:
            # Parse result to extract paper info
            paper_info = {
                "source": source_name,
                "title": "",
                "authors": [],
                "abstract": "",
                "url": "",
                "year": None,
                "citations": None
            }
            
            # Extract title and content
            if ":" in result:
                parts = result.split(":", 1)
                paper_info["title"] = parts[0].strip()
                paper_info["abstract"] = parts[1].strip() if len(parts) > 1 else ""
            
            # Extract URL from brackets
            url_match = re.search(r'\[([^\]]+)\]', result)
            if url_match:
                paper_info["url"] = f"https://{url_match.group(1)}"
            
            # Extract year
            year_match = re.search(r'20\d{2}|19\d{2}', result)
            if year_match:
                paper_info["year"] = int(year_match.group())
            
            academic_results.append(paper_info)
    
    return academic_results[:count]

def fact_check(claim: str) -> Dict[str, Any]:
    """Fact-check a claim using multiple sources"""
    fact_check_results = {
        "claim": claim,
        "sources": [],
        "verdict": "unverified",
        "confidence": 0.0,
        "evidence": []
    }
    
    # Search fact-checking sites
    fact_check_query = f'"{claim}" site:snopes.com OR site:factcheck.org OR site:politifact.com'
    results = search_web(fact_check_query, count=5)
    
    # Also search for general evidence
    evidence_query = f'{claim} evidence facts'
    evidence_results = search_web(evidence_query, count=5)
    
    # Analyze results
    supporting = 0
    refuting = 0
    
    all_results = results + evidence_results
    for result in all_results:
        lower_result = result.lower()
        
        # Look for verdict indicators
        if any(word in lower_result for word in ["true", "correct", "verified", "confirmed"]):
            supporting += 1
        elif any(word in lower_result for word in ["false", "incorrect", "debunked", "myth"]):
            refuting += 1
        
        fact_check_results["evidence"].append(result)
    
    # Determine verdict
    total_evidence = supporting + refuting
    if total_evidence > 0:
        if supporting > refuting:
            fact_check_results["verdict"] = "likely true"
            fact_check_results["confidence"] = supporting / total_evidence
        elif refuting > supporting:
            fact_check_results["verdict"] = "likely false"
            fact_check_results["confidence"] = refuting / total_evidence
        else:
            fact_check_results["verdict"] = "disputed"
            fact_check_results["confidence"] = 0.5
    
    return fact_check_results

def extract_plain_answer(results: List[str]) -> Optional[str]:
    """Extract a plain, comprehensive answer from search results"""
    if not results:
        return None
    
    # Filter out noise and combine relevant information
    clean_results = []
    seen_content = set()
    
    for result in results:
        # Remove HTML and clean
        plain = clean_html(result)
        
        # Skip if too short or duplicate
        if len(plain.strip()) < 30:
            continue
        
        # Simple deduplication
        content_hash = hash(plain.lower().strip())
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)
        
        clean_results.append(plain.strip())
    
    if not clean_results:
        return None
    
    # For single result, return as-is
    if len(clean_results) == 1:
        return clean_results[0]
    
    # For multiple results, create a synthesized answer
    # Prioritize informational content
    info_box = next((r for r in clean_results if r.startswith("[INFO]")), None)
    if info_box:
        return info_box.replace("[INFO]", "").strip()
    
    # Otherwise, combine the most relevant parts
    combined = []
    for result in clean_results[:3]:  # Use top 3 results
        # Extract the most informative sentence
        sentences = result.split(". ")
        if sentences:
            # Prefer sentences with numbers, dates, or definitions
            informative = sorted(sentences, 
                               key=lambda s: len(re.findall(r'\d+|is |are |was |were |means |refers', s)),
                               reverse=True)
            if informative:
                combined.append(informative[0])
    
    return ". ".join(combined) if combined else clean_results[0]

def clean_html(text: str) -> str:
    """Enhanced HTML cleaning"""
    if not text:
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    
    return text.strip()

def monitor_topic(topic: str, interval_hours: int = 24) -> Dict[str, Any]:
    """Monitor a topic for changes over time"""
    monitor_file = f"{CACHE_DIR}/monitor_{quote(topic)}.json"
    
    # Load previous monitoring data
    try:
        with open(monitor_file, "r") as f:
            monitor_data = json.load(f)
    except:
        monitor_data = {
            "topic": topic,
            "checks": [],
            "changes_detected": []
        }
    
    # Perform current check
    current_results = search_web(f"{topic} latest news", count=10, search_type="news")
    current_check = {
        "timestamp": datetime.now().isoformat(),
        "results": current_results,
        "hash": hash(str(current_results))
    }
    
    # Compare with previous check
    if monitor_data["checks"]:
        last_check = monitor_data["checks"][-1]
        if last_check["hash"] != current_check["hash"]:
            # Changes detected
            change_info = {
                "timestamp": datetime.now().isoformat(),
                "type": "content_change",
                "summary": f"New developments in {topic}"
            }
            monitor_data["changes_detected"].append(change_info)
    
    # Add current check
    monitor_data["checks"].append(current_check)
    
    # Keep only recent checks (last 30 days)
    cutoff_date = datetime.now().timestamp() - (30 * 24 * 3600)
    monitor_data["checks"] = [
        check for check in monitor_data["checks"]
        if datetime.fromisoformat(check["timestamp"]).timestamp() > cutoff_date
    ]
    
    # Save monitoring data
    with open(monitor_file, "w") as f:
        json.dump(monitor_data, f, indent=2)
    
    return {
        "topic": topic,
        "last_check": current_check["timestamp"],
        "total_checks": len(monitor_data["checks"]),
        "changes_detected": len(monitor_data["changes_detected"]),
        "recent_changes": monitor_data["changes_detected"][-5:]
    }

def _get_cached_search(cache_key: str, max_age_hours: int = 24) -> Optional[List[str]]:
    """Get cached search results if available and fresh"""
    cache_file = f"{CACHE_DIR}/{quote(cache_key)}.json"
    
    try:
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
        
        # Check age
        cached_time = datetime.fromisoformat(cache_data["timestamp"])
        age_hours = (datetime.now() - cached_time).total_seconds() / 3600
        
        if age_hours < max_age_hours:
            return cache_data["results"]
    except:
        pass
    
    return None

def _cache_search_result(cache_key: str, results: List[str]):
    """Cache search results"""
    cache_file = f"{CACHE_DIR}/{quote(cache_key)}.json"
    
    cache_data = {
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    try:
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error caching search results: {e}")

def analyze_search_trends(queries: List[str]) -> Dict[str, Any]:
    """Analyze trends in search queries"""
    trends = {
        "total_queries": len(queries),
        "unique_queries": len(set(queries)),
        "common_terms": {},
        "query_categories": {},
        "temporal_patterns": {}
    }
    
    # Extract common terms
    all_terms = []
    for query in queries:
        terms = query.lower().split()
        all_terms.extend(terms)
    
    # Count term frequency
    from collections import Counter
    term_counts = Counter(all_terms)
    
    # Filter out common words
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "is", "was", "are", "were"}
    trends["common_terms"] = {
        term: count for term, count in term_counts.most_common(20)
        if term not in stop_words and len(term) > 2
    }
    
    # Categorize queries
    categories = {
        "technical": ["code", "programming", "algorithm", "software", "debug"],
        "research": ["study", "research", "paper", "academic", "theory"],
        "news": ["latest", "news", "current", "today", "recent"],
        "how_to": ["how to", "tutorial", "guide", "learn", "example"],
        "definition": ["what is", "define", "meaning", "definition", "explain"]
    }
    
    for query in queries:
        query_lower = query.lower()
        categorized = False
        
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                trends["query_categories"][category] = trends["query_categories"].get(category, 0) + 1
                categorized = True
                break
        
        if not categorized:
            trends["query_categories"]["other"] = trends["query_categories"].get("other", 0) + 1
    
    return trends

class WebResearchAssistant:
    """Advanced web research capabilities"""
    
    def __init__(self):
        self.search_history = []
        self.research_sessions = {}
    
    async def conduct_research(self, topic: str, depth: str = "moderate") -> Dict[str, Any]:
        """Conduct comprehensive research on a topic"""
        research_id = f"research_{datetime.now().timestamp()}"
        
        research_plan = {
            "shallow": {"searches": 3, "sources": 5},
            "moderate": {"searches": 5, "sources": 10},
            "deep": {"searches": 10, "sources": 20}
        }
        
        plan = research_plan.get(depth, research_plan["moderate"])
        
        research_results = {
            "id": research_id,
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "depth": depth,
            "findings": [],
            "summary": "",
            "sources": [],
            "key_facts": [],
            "related_topics": []
        }
        
        # Phase 1: Initial broad search
        initial_results = search_web(topic, count=plan["searches"])
        research_results["findings"].extend(initial_results)
        
        # Phase 2: Identify subtopics and search them
        subtopics = self._extract_subtopics(initial_results)
        for subtopic in subtopics[:3]:
            sub_results = search_web(f"{topic} {subtopic}", count=3)
            research_results["findings"].extend(sub_results)
        
        # Phase 3: Fact-check key claims
        key_claims = self._extract_claims(research_results["findings"])
        for claim in key_claims[:3]:
            fact_result = fact_check(claim)
            research_results["key_facts"].append(fact_result)
        
        # Phase 4: Academic search if applicable
        if depth == "deep":
            academic_results = search_academic(topic, count=5)
            research_results["sources"].extend(academic_results)
        
        # Phase 5: Synthesize findings
        research_results["summary"] = self._synthesize_research(research_results)
        research_results["related_topics"] = subtopics
        
        # Store research session
        self.research_sessions[research_id] = research_results
        
        return research_results
    
    def _extract_subtopics(self, results: List[str]) -> List[str]:
        """Extract subtopics from search results"""
        # Simple extraction based on common phrases
        subtopics = []
        
        combined_text = " ".join(results).lower()
        
        # Look for phrases that indicate subtopics
        patterns = [
            r"including ([\w\s]+)",
            r"such as ([\w\s]+)",
            r"types of ([\w\s]+)",
            r"related to ([\w\s]+)",
            r"aspects of ([\w\s]+)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, combined_text)
            subtopics.extend(matches)
        
        # Clean and deduplicate
        cleaned_subtopics = []
        for topic in subtopics:
            cleaned = topic.strip()
            if len(cleaned) > 3 and len(cleaned) < 50:
                cleaned_subtopics.append(cleaned)
        
        return list(set(cleaned_subtopics))[:10]
    
    def _extract_claims(self, results: List[str]) -> List[str]:
        """Extract factual claims from results"""
        claims = []
        
        # Patterns that indicate factual claims
        claim_patterns = [
            r"([\w\s]+ (?:is|are|was|were) [\w\s]+)",
            r"([\w\s]+ (?:has|have|had) [\w\s]+)",
            r"([\w\s]+ (?:causes|caused) [\w\s]+)",
            r"([\d]+ percent [\w\s]+)",
            r"([\w\s]+ (?:increased|decreased) by [\d]+)"
        ]
        
        for result in results:
            for pattern in claim_patterns:
                matches = re.findall(pattern, result)
                claims.extend(matches)
        
        # Filter and clean claims
        cleaned_claims = []
        for claim in claims:
            cleaned = claim.strip()
            if 10 < len(cleaned) < 200:
                cleaned_claims.append(cleaned)
        
        return list(set(cleaned_claims))[:10]
    
    def _synthesize_research(self, research_data: Dict[str, Any]) -> str:
        """Synthesize research findings into a summary"""
        findings = research_data["findings"]
        facts = research_data["key_facts"]
        
        summary_parts = []
        
        # Introduction
        summary_parts.append(f"Research on '{research_data['topic']}' yielded {len(findings)} findings.")
        
        # Key verified facts
        verified_facts = [f for f in facts if f["verdict"] == "likely true"]
        if verified_facts:
            summary_parts.append(f"Verified facts: {len(verified_facts)} claims confirmed.")
        
        # Main themes (simplified extraction)
        all_text = " ".join(findings)
        word_count = len(all_text.split())
        summary_parts.append(f"Analysis covered {word_count} words of content.")
        
        # Related topics
        if research_data["related_topics"]:
            summary_parts.append(f"Related topics identified: {', '.join(research_data['related_topics'][:3])}")
        
        return " ".join(summary_parts)
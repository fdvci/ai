import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

# Make sklearn optional:
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

@dataclass
class Memory:
    """Simple data container representing a single memory entry."""
    id: str
    timestamp: str
    speaker: str
    type: str
    content: str
    embedding_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


logger = logging.getLogger(__name__)

class VectorMemory:
    def __init__(self, persist_directory: str = "data/vector_db", max_memories: int = 50000):
        """Initialize vector memory with persistence and bounds checking"""
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.max_memories = max_memories
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create embedding function
        try:
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            # Fallback to default embeddings if OpenAI fails
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Create collections for different memory types
        self.collections = {
            "general": self._get_or_create_collection("nova_general_memory"),
            "facts": self._get_or_create_collection("nova_facts"),
            "insights": self._get_or_create_collection("nova_insights"),
            "emotions": self._get_or_create_collection("nova_emotions"),
            "goals": self._get_or_create_collection("nova_goals")
        }
        
        # Memory statistics
        self.stats = self.load_stats()
        
        # Check memory limits on init - CALL THE METHOD HERE
        self._check_memory_limits()

    def _check_memory_limits(self):
        """Check and enforce memory limits"""
        try:
            total_memories = sum(
                collection.count() for collection in self.collections.values()
            )
            
            if total_memories > self.max_memories:
                logger.warning(f"Memory limit exceeded: {total_memories}/{self.max_memories}")
                # Trigger cleanup
                deleted = self.cleanup_old_memories(days=30)
                logger.info(f"Cleaned up {deleted} old memories")
        except Exception as e:
            logger.error(f"Error checking memory limits: {e}")
        
    def _get_or_create_collection(self, name: str):
        """Get or create a collection with error handling"""
        try:
            return self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_function,
                metadata={"created_at": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.error(f"Error creating collection {name}: {e}")
            # Try to get existing collection
            try:
                return self.client.get_collection(name=name)
            except:
                raise
    
    def load_stats(self) -> Dict[str, Any]:
        """Load memory statistics"""
        stats_file = self.persist_directory / "memory_stats.json"
        try:
            with open(stats_file, "r") as f:
                return json.load(f)
        except:
            return {
                "total_memories": 0,
                "memories_by_type": {},
                "last_cleanup": None,
                "query_count": 0,
                "avg_query_time": 0
            }
    
    def save_stats(self):
        """Save memory statistics"""
        stats_file = self.persist_directory / "memory_stats.json"
        with open(stats_file, "w") as f:
            json.dump(self.stats, f, indent=2)
    
    def store(self, text: str, memory_id: str, memory_type: str = "general", 
          metadata: Optional[Dict[str, Any]] = None):
        """Store memory with type-specific collection, metadata, and bounds checking"""
        try:
            # Check memory limits before storing
            if self.get_total_memory_count() >= self.max_memories:
                logger.warning("Memory limit reached, triggering cleanup")
                self.cleanup_old_memories(days=7)  # Aggressive cleanup
            
            # Select appropriate collection
            collection_type = memory_type if memory_type in self.collections else "general"
            collection = self.collections[collection_type]
            
            # Prepare metadata
            full_metadata = {
                "timestamp": datetime.now().isoformat(),
                "type": memory_type,
                "length": len(text)
            }
            if metadata:
                full_metadata.update(metadata)
            
            # Store in collection
            collection.add(
                documents=[text],
                ids=[memory_id],
                metadatas=[full_metadata]
            )
            
            # Update statistics
            self.stats["total_memories"] += 1
            if memory_type not in self.stats["memories_by_type"]:
                self.stats["memories_by_type"][memory_type] = 0
            self.stats["memories_by_type"][memory_type] += 1
            self.save_stats()
            
            logger.debug(f"Stored memory {memory_id} of type {memory_type}")
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise MemoryError(f"Failed to store memory {memory_id}: {e}")

    def get_total_memory_count(self) -> int:
        """Get total count of memories across all collections"""
        try:
            return sum(collection.count() for collection in self.collections.values())
        except Exception as e:
            logger.error(f"Error counting memories: {e}")
            return 0
                
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
    
    def retrieve(self, query: str, k: int = 5, memory_types: Optional[List[str]] = None,
                threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Retrieve memories with advanced filtering and ranking"""
        start_time = datetime.now()
        results = []
        
        try:
            # Determine which collections to search
            if memory_types:
                search_collections = [(t, self.collections[t]) for t in memory_types 
                                    if t in self.collections]
            else:
                search_collections = list(self.collections.items())
            
            # Search each relevant collection
            for coll_type, collection in search_collections:
                try:
                    # Check if collection has items
                    collection_count = collection.count()
                    if collection_count == 0:
                        continue
                    
                    # Ensure we don't query for more than available
                    n_results = min(k, collection_count)
                    
                    query_results = collection.query(
                        query_texts=[query],
                        n_results=n_results,
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    # Process results
                    if query_results["documents"] and query_results["documents"][0]:
                        for i, doc in enumerate(query_results["documents"][0]):
                            distance = query_results["distances"][0][i] if query_results["distances"] else 1.0
                            similarity = 1.0 - distance  # Convert distance to similarity
                            
                            if similarity >= threshold:
                                results.append({
                                    "content": doc,
                                    "metadata": query_results["metadatas"][0][i] if query_results["metadatas"] else {},
                                    "similarity": similarity,
                                    "collection": coll_type
                                })
                
                except Exception as e:
                    logger.error(f"Error querying collection {coll_type}: {e}")
            
            # Sort by similarity
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Update query statistics
            query_time = (datetime.now() - start_time).total_seconds()
            self.stats["query_count"] += 1
            self.stats["avg_query_time"] = (
                (self.stats["avg_query_time"] * (self.stats["query_count"] - 1) + query_time) 
                / self.stats["query_count"]
            )
            self.save_stats()
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    def update(self, memory_id: str, new_text: str, memory_type: str = "general",
               metadata: Optional[Dict[str, Any]] = None):
        """Update existing memory"""
        try:
            collection = self.collections.get(memory_type, self.collections["general"])
            
            # Delete old version
            collection.delete(ids=[memory_id])
            
            # Add updated version
            self.store(new_text, memory_id, memory_type, metadata)
            
            logger.info(f"Updated memory {memory_id}")
            
        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}")
    
    def delete(self, memory_id: str, memory_type: Optional[str] = None):
        """Delete memory from specified or all collections"""
        try:
            if memory_type and memory_type in self.collections:
                self.collections[memory_type].delete(ids=[memory_id])
                logger.info(f"Deleted memory {memory_id} from {memory_type}")
            else:
                # Delete from all collections
                for coll_type, collection in self.collections.items():
                    try:
                        collection.delete(ids=[memory_id])
                    except:
                        pass
                logger.info(f"Deleted memory {memory_id} from all collections")
            
            # Update stats
            self.stats["total_memories"] = max(0, self.stats["total_memories"] - 1)
            self.save_stats()
            
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
    
    def semantic_search(self, query: str, filters: Dict[str, Any], k: int = 10) -> List[Dict[str, Any]]:
        """Advanced semantic search with metadata filtering"""
        results = []
        
        for coll_type, collection in self.collections.items():
            try:
                # Build where clause from filters
                where_clause = {}
                if "after_date" in filters:
                    where_clause["timestamp"] = {"$gte": filters["after_date"]}
                if "speaker" in filters:
                    where_clause["speaker"] = filters["speaker"]
                if "min_similarity" in filters:
                    # This will be handled post-query
                    pass
                
                query_results = collection.query(
                    query_texts=[query],
                    n_results=k,
                    where=where_clause if where_clause else None,
                    include=["documents", "metadatas", "distances"]
                )
                
                # Process and filter results
                for i, doc in enumerate(query_results["documents"][0] if query_results["documents"] else []):
                    distance = query_results["distances"][0][i] if query_results["distances"] else 1.0
                    similarity = 1.0 - distance
                    
                    if "min_similarity" not in filters or similarity >= filters["min_similarity"]:
                        results.append({
                            "content": doc,
                            "metadata": query_results["metadatas"][0][i] if query_results["metadatas"] else {},
                            "similarity": similarity,
                            "collection": coll_type
                        })
                        
            except Exception as e:
                logger.error(f"Error in semantic search for {coll_type}: {e}")
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]
    
    def find_similar_memories(self, memory_id: str, memory_type: str = "general", 
                            k: int = 5) -> List[Dict[str, Any]]:
        """Find memories similar to a given memory"""
        try:
            collection = self.collections.get(memory_type, self.collections["general"])
            
            # Get the memory content
            memory_data = collection.get(ids=[memory_id], include=["documents"])
            if not memory_data["documents"]:
                return []
            
            memory_content = memory_data["documents"][0]
            
            # Find similar memories (excluding the original)
            similar = self.retrieve(memory_content, k=k+1, memory_types=[memory_type])
            
            # Filter out the original memory
            return [m for m in similar if m.get("metadata", {}).get("id") != memory_id][:k]
            
        except Exception as e:
            logger.error(f"Error finding similar memories: {e}")
            return []
    
    def cluster_memories(self, memory_type: str = "general", n_clusters: int = 5) -> Dict[int, List[str]]:
        """Cluster memories to identify themes"""
        try:
            collection = self.collections.get(memory_type, self.collections["general"])
            
            # Get all memories
            all_data = collection.get(include=["embeddings", "ids"])
            if not all_data["embeddings"]:
                return {}
            
            embeddings = np.array(all_data["embeddings"])
            ids = all_data["ids"]
            
            # Simple k-means clustering
            from sklearn.cluster import KMeans
            
            n_clusters = min(n_clusters, len(embeddings))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            # Group by cluster
            cluster_groups = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(ids[i])
            
            return cluster_groups
            
        except Exception as e:
            logger.error(f"Error clustering memories: {e}")
            return {}
    
    def cleanup_old_memories(self, days: int = 30, keep_important: bool = True):
        """Clean up old memories while preserving important ones"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            deleted_count = 0
            
            for coll_type, collection in self.collections.items():
                # Get old memories
                where_clause = {"timestamp": {"$lt": cutoff_date}}
                
                if keep_important:
                    # Don't delete facts, insights, or goals
                    if coll_type in ["facts", "insights", "goals"]:
                        continue
                
                # Get memories to delete
                old_memories = collection.get(
                    where=where_clause,
                    include=["ids", "metadatas"]
                )
                
                if old_memories["ids"]:
                    # Filter out important memories
                    ids_to_delete = []
                    for i, memory_id in enumerate(old_memories["ids"]):
                        metadata = old_memories["metadatas"][i] if old_memories["metadatas"] else {}
                        
                        # Keep memories marked as important
                        if not metadata.get("important", False):
                            ids_to_delete.append(memory_id)
                    
                    if ids_to_delete:
                        collection.delete(ids=ids_to_delete)
                        deleted_count += len(ids_to_delete)
                        logger.info(f"Deleted {len(ids_to_delete)} old memories from {coll_type}")
            
            # Update stats
            self.stats["total_memories"] -= deleted_count
            self.stats["last_cleanup"] = datetime.now().isoformat()
            self.save_stats()
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old memories: {e}")
            return 0
    
    def export_memories(self, output_file: str, memory_types: Optional[List[str]] = None):
        """Export memories to JSON file"""
        try:
            export_data = {
                "export_date": datetime.now().isoformat(),
                "stats": self.stats,
                "memories": {}
            }
            
            collections_to_export = memory_types if memory_types else list(self.collections.keys())
            
            for coll_type in collections_to_export:
                if coll_type in self.collections:
                    collection = self.collections[coll_type]
                    all_data = collection.get(include=["documents", "metadatas", "ids"])
                    
                    export_data["memories"][coll_type] = []
                    for i in range(len(all_data["ids"])):
                        export_data["memories"][coll_type].append({
                            "id": all_data["ids"][i],
                            "content": all_data["documents"][i] if all_data["documents"] else "",
                            "metadata": all_data["metadatas"][i] if all_data["metadatas"] else {}
                        })
            
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported memories to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting memories: {e}")
            return False
    
    def import_memories(self, input_file: str, merge: bool = True):
        """Import memories from JSON file"""
        try:
            with open(input_file, "r") as f:
                import_data = json.load(f)
            
            if not merge:
                # Clear existing memories
                for collection in self.collections.values():
                    collection.delete(collection.get()["ids"])
            
            # Import memories
            imported_count = 0
            for coll_type, memories in import_data.get("memories", {}).items():
                if coll_type in self.collections:
                    for memory in memories:
                        self.store(
                            text=memory["content"],
                            memory_id=memory["id"],
                            memory_type=coll_type,
                            metadata=memory.get("metadata", {})
                        )
                        imported_count += 1
            
            logger.info(f"Imported {imported_count} memories from {input_file}")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing memories: {e}")
            return 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics"""
        stats = self.stats.copy()
        
        # Add current collection sizes
        stats["current_sizes"] = {}
        for coll_type, collection in self.collections.items():
            try:
                stats["current_sizes"][coll_type] = collection.count()
            except:
                stats["current_sizes"][coll_type] = 0
        
        # Calculate memory health metrics
        total_memories = sum(stats["current_sizes"].values())
        stats["memory_health"] = {
            "total_memories": total_memories,
            "diversity_score": len([s for s in stats["current_sizes"].values() if s > 0]) / len(self.collections),
            "avg_memories_per_type": total_memories / len(self.collections) if self.collections else 0
        }
        
        return stats
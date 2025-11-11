"""Example usage of GraphStore for Neo4j operations.

This example demonstrates how to use the GraphStore class from the project:
1. Connect to Neo4j database using the GraphStore class
2. Create simple sample data
3. Query the data using Cypher queries through the GraphStore
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.stores.graph_store import TechGraphStore
from src.utils.logger import logger


def main():
    """Main function demonstrating GraphStore usage."""
    
    try:
        # Initialize GraphStore (uses environment variables or defaults)
        logger.info("Connecting to Neo4j using GraphStore...")
        graph_store = TechGraphStore(
            uri="neo4j://127.0.0.1:7687",
            username="neo4j", 
            password="password",
            database="neo4j"
        )
        
        # Example 1: Clear existing data (optional)
        logger.info("Clearing existing data...")
        graph_store.graph.query("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared")
        
        # Example 2: Create simple sample data
        logger.info("Creating sample technology radar data...")
        
        # Create sample technologies
        technologies = [
            {
                'name': 'React',
                'quadrant': 'Languages and Frameworks',
                'ring': 'Adopt',
                'description': 'React continues to be our go-to choice for building user interfaces.'
            },
            {
                'name': 'Kubernetes',
                'quadrant': 'Platforms',
                'ring': 'Adopt',
                'description': 'Kubernetes has become the de facto standard for container orchestration.'
            },
            {
                'name': 'GraphQL',
                'quadrant': 'Techniques',
                'ring': 'Trial',
                'description': 'GraphQL shows promise for API development but needs more evaluation.'
            },
            {
                'name': 'TypeScript',
                'quadrant': 'Languages and Frameworks',
                'ring': 'Adopt',
                'description': 'TypeScript provides excellent type safety for JavaScript applications.'
            },
            {
                'name': 'Docker',
                'quadrant': 'Platforms',
                'ring': 'Adopt',
                'description': 'Docker containers are essential for modern application deployment.'
            }
        ]
        
        # Insert technologies
        for tech in technologies:
            query = """
            CREATE (t:Technology {
                name: $name,
                quadrant: $quadrant,
                ring: $ring,
                description: $description
            })
            """
            graph_store.graph.query(query, tech)
        
        logger.info(f"Created {len(technologies)} technology nodes")
        
        # Example 3: Query the data
        logger.info("Querying the graph...")
        
        # Get all technologies
        all_techs = graph_store.graph.query("MATCH (t:Technology) RETURN t.name as name, t.quadrant as quadrant, t.ring as ring")
        logger.info(f"Total technologies: {len(all_techs)}")
        
        # Query by quadrant
        frameworks = graph_store.graph.query(
            "MATCH (t:Technology) WHERE t.quadrant = 'Languages and Frameworks' RETURN t.name as name, t.ring as ring"
        )
        logger.info(f"Languages and Frameworks: {len(frameworks)} technologies")
        for tech in frameworks:
            logger.info(f"  - {tech['name']} ({tech['ring']})")
        
        # Query by ring
        adopt_techs = graph_store.graph.query(
            "MATCH (t:Technology) WHERE t.ring = 'Adopt' RETURN t.name as name, t.quadrant as quadrant"
        )
        logger.info(f"Adopt ring: {len(adopt_techs)} technologies")
        for tech in adopt_techs:
            logger.info(f"  - {tech['name']} ({tech['quadrant']})")
        
        # Search for specific technology
        react_search = graph_store.graph.query(
            "MATCH (t:Technology) WHERE t.name CONTAINS 'React' RETURN t.name as name, t.description as description"
        )
        logger.info(f"Technologies containing 'React': {len(react_search)}")
        for tech in react_search:
            logger.info(f"  - {tech['name']}: {tech['description']}")
        
        # Example 4: Get simple statistics
        logger.info("Getting statistics...")
        
        stats = {
            'total_nodes': graph_store.graph.query("MATCH (n) RETURN count(n) as count")[0]['count'],
            'total_technologies': graph_store.graph.query("MATCH (t:Technology) RETURN count(t) as count")[0]['count'],
            'adopt_count': graph_store.graph.query("MATCH (t:Technology) WHERE t.ring = 'Adopt' RETURN count(t) as count")[0]['count'],
            'trial_count': graph_store.graph.query("MATCH (t:Technology) WHERE t.ring = 'Trial' RETURN count(t) as count")[0]['count']
        }
        
        logger.info("Graph Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Example 5: Show technologies by quadrant
        quadrants = ['Languages and Frameworks', 'Platforms', 'Techniques', 'Tools']
        for quadrant in quadrants:
            count = graph_store.graph.query(
                "MATCH (t:Technology) WHERE t.quadrant = $quadrant RETURN count(t) as count",
                {'quadrant': quadrant}
            )[0]['count']
            if count > 0:
                logger.info(f"{quadrant}: {count} technologies")
        
        # Close connection
        graph_store.close()
        logger.info("Connection closed successfully")
        
    except Exception as e:
        logger.error(f"Error in GraphStore example: {e}")
        logger.info("Make sure Neo4j is running and the connection parameters are correct.")
        logger.info("You can start Neo4j using Docker:")
        logger.info("  docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
        logger.info("Or set environment variables: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE")


if __name__ == "__main__":
    main()

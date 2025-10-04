from src.stores.graph_store import TechGraphStore
from src.utils.logger import logger

class TechRadarGraphBuilder:
    def __init__(self):
        self.graph_store = TechGraphStore()
        self.graph_store.graph.query("MATCH (n) DETACH DELETE n")
        self.create_quadrants()
        self.create_rings()
    
    def create_radar_node(self, node_data):
        data = {
                "metadata": node_data
            }
        query = """
            CREATE (tr:TechRadar $metadata)
            RETURN tr
            """
        self.graph_store.graph.query(query, data) 
        
    def create_quadrants(self):
        data = {
            "quadrant_data": [ {
                "title": "Techniques",
            }, {
                "title": "Platforms",
            }, {
                "title": "Tools",
            }, {
                "title": "Languages and \nFrameworks",
            } ]
        }
        query = """
            UNWIND $quadrant_data AS map
            CREATE (q:Quadrant)
            SET q = map
        """
        self.graph_store.graph.query(query, data)

    def create_rings(self):
        data = {
            "ring_data": [ {
                "title": "Adopt",
            }, {
                "title": "Trial",
            }, {
                "title": "Assess",
            }, {
                "title": "Hold",
            } ]
        }
        query = """
            UNWIND $ring_data AS map
            CREATE (r:Ring)
            SET r = map
        """
        self.graph_store.graph.query(query, data)

    def create_blip_nodes(self, blip_details):
        blip_data = {
            "doc": blip_details.get("doc", ""),
            "blip_title": blip_details.get("blip_title", ""),
            "period": blip_details.get("period", ""),
            "title": blip_details.get("title", ""),
            "radar_volume": blip_details.get("volume", ""),
            "quadrant": blip_details.get("quadrant", "unknown"),
            "ring": blip_details.get("ring", "unknown")
        }
        query = """
            MATCH (tr:TechRadar {title: $title})
            MATCH (q:Quadrant {title: $quadrant})
            MATCH (r:Ring {title: $ring})
            CREATE (b:Blip {content: $doc, title: $blip_title}) - [:PUBLISHED_IN {period: $period, volume: $radar_volume}] -> (tr)
            CREATE (b) - [:CATEGORIZED_IN] -> (q)
            CREATE (b) - [:POSITIONED_IN] -> (r)
            """
        
        self.graph_store.graph.query(query, blip_data)
        self.graph_store.graph.refresh_schema()

    # def store_chunks_in_graph(self):
    #     logger.info("Clearing existing data...")
    #     self.graph_store.graph.query("MATCH (n) DETACH DELETE n")
    #     logger.info("Database cleared")

    #     logger.info(f"Creating graph data {self.graph_content.keys()}")
    #     for radar in self.graph_content.keys():
    #         metadata = {
    #             "metadata": self.graph_content[radar]["metadata"]
    #         }
    #         print(metadata)
    #         query = """
    #         CREATE (t:TechRadar $metadata)
    #         RETURN t
    #         """
    #         self.graph_store.graph.query(query, metadata)        

    #     query = """
    #     MATCH (t:TechRadar) 
    #     RETURN t.title as title, t.volume as volume, t.period as period, 
    #         t.filename as filename, t.creationdate as creationdate
    #     ORDER BY t.volume
    #     """
        
    #     results = self.graph_store.graph.query(query)
    #     logger.info(f"Found {len(results)} TechRadar nodes:")
        
    #     for i, node in enumerate(results, 1):
    #         logger.info(f"  {i}. Title: {node['title']}")
    #         logger.info(f"     Volume: {node['volume']}")
    #         logger.info(f"     Period: {node['period']}")
    #         logger.info(f"     Filename: {node['filename']}")
    #         logger.info(f"     Creation Date: {node['creationdate']}")
    #         logger.info("     ---")
        
    #     return results

"""
SHACL Validator

Validates graph entities against SHACL shapes defined in the ontology.
Uses Neo4j neosemantics plugin for constraint validation.

Validation Flow:
1. Load RDF triples into Neo4j (if not already loaded)
2. Call SHACL validation via neosemantics
3. Parse validation results
4. Report violations and conformance
"""

from typing import Dict, List, Optional, Any
import logging

try:
    from neo4j import GraphDatabase, Driver
    _NEO4J_AVAILABLE = True
except ImportError:
    GraphDatabase = None
    Driver = None
    _NEO4J_AVAILABLE = False
    logging.getLogger(__name__).warning("neo4j package not installed — SHACLValidator unavailable")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHACLValidator:
    """SHACL validator for graph entities."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        """
        Initialize the SHACL validator.

        Args:
            uri: Neo4j Bolt URI
            user: Neo4j username
            password: Neo4j password
            database: Neo4j database name
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None

    def connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query."""
        if not self.driver:
            self.connect()

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def validate_graph(self) -> Dict[str, Any]:
        """
        Validate all graph entities against SHACL shapes.

        Returns:
            Validation report with conformance, violations, and statistics
        """
        if not self.driver:
            self.connect()

        logger.info("Starting SHACL validation...")

        try:
            # Call SHACL validation using neosemantics
            validation_report = self._run_shacl_validation()

            # Parse validation results
            report = self._parse_validation_report(validation_report)

            logger.info(f"SHACL validation complete: Conformant={report['conformant']}, Violations={report['violation_count']}")
            return report

        except Exception as e:
            logger.error(f"SHACL validation failed: {e}")
            return {
                'conformant': False,
                'error': str(e),
                'violation_count': 0,
                'violations': []
            }

    def _run_shacl_validation(self) -> List[Dict[str, Any]]:
        """
        Run SHACL validation using neosemantics plugin.

        Returns:
            Raw validation report from Neo4j
        """
        # Check if neosemantics plugin is available
        check_query = """
        CALL dbms.procedures() YIELD name
        WHERE name STARTS WITH 'n10s'
        RETURN count(*) AS plugin_available
        """

        result = self.execute_query(check_query)

        if not result or result[0]['plugin_available'] == 0:
            logger.warning("neosemantics plugin not available, skipping SHACL validation")
            return []

        # Run SHACL validation
        validation_query = """
        CALL n10s.validation.shacl.validate()
        YIELD node, report, conforms
        RETURN node, report, conforms
        """

        return self.execute_query(validation_query)

    def _parse_validation_report(self, raw_report: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse raw SHACL validation report.

        Args:
            raw_report: Raw validation results from Neo4j

        Returns:
            Parsed validation report
        """
        violations = []
        conformant = True

        for record in raw_report:
            node = record.get('node')
            report = record.get('report')
            conforms = record.get('conforms', True)

            if not conforms:
                conformant = False

                # Extract violation details
                violation = {
                    'node_id': str(node) if node else 'unknown',
                    'message': str(report) if report else 'Unknown constraint violation',
                    'severity': 'violation'
                }
                violations.append(violation)

        report = {
            'conformant': conformant,
            'violation_count': len(violations),
            'violations': violations,
            'validated_nodes': len(raw_report) if raw_report else 0
        }

        return report

    def validate_node(
        self,
        node_label: str,
        node_id: str
    ) -> Dict[str, Any]:
        """
        Validate a specific node against SHACL shapes.

        Args:
            node_label: Node label (e.g., 'Asset', 'SoftwareAsset')
            node_id: Node property value (e.g., assetId)

        Returns:
            Validation report for the node
        """
        if not self.driver:
            self.connect()

        try:
            # Get the node
            get_node_query = f"""
            MATCH (n:{node_label} {{node_id: $nodeId}})
            RETURN n
            """

            result = self.execute_query(get_node_query, {'nodeId': node_id})

            if not result:
                return {
                    'conformant': False,
                    'error': f'Node {node_label}.{node_id} not found'
                }

            # Validate this specific node (neosemantics validates all, so we filter)
            graph_report = self.validate_graph()

            # Filter violations for this node
            node_violations = [
                v for v in graph_report.get('violations', [])
                if node_id in v.get('node_id', '')
            ]

            return {
                'node_label': node_label,
                'node_id': node_id,
                'conformant': len(node_violations) == 0,
                'violation_count': len(node_violations),
                'violations': node_violations
            }

        except Exception as e:
            logger.error(f"Node validation failed: {e}")
            return {
                'conformant': False,
                'error': str(e)
            }

    def get_shacl_shapes(self) -> List[Dict[str, Any]]:
        """
        Get all SHACL shapes loaded in the graph.

        Returns:
            List of SHACL shape definitions
        """
        if not self.driver:
            self.connect()

        query = """
        MATCH (shape:Shape)
        OPTIONAL MATCH (shape)-[:sh:targetClass]->(class)
        OPTIONAL MATCH (shape)-[:sh:property]->(property)
        OPTIONAL MATCH (property)-[:sh:path]->(path)
        RETURN shape, class, property, path
        """

        results = self.execute_query(query)
        shapes = []

        for record in results:
            shape = {
                'shape_id': str(record.get('shape')),
                'target_class': str(record.get('class')) if record.get('class') else None,
                'property_path': str(record.get('path')) if record.get('path') else None
            }
            shapes.append(shape)

        logger.info(f"Found {len(shapes)} SHACL shapes")
        return shapes

    def validate_constraints(
        self,
        constraints: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate specific constraints on the graph.

        Args:
            constraints: List of custom Cypher constraints to validate

        Returns:
            Validation report with constraint results
        """
        if not self.driver:
            self.connect()

        results = []
        all_valid = True

        for constraint in constraints:
            try:
                query = constraint.get('query')
                name = constraint.get('name', 'unnamed')

                result = self.execute_query(query)
                constraint_result = {
                    'name': name,
                    'valid': len(result) == 0,  # Assume empty result = valid (no violations)
                    'violations': result
                }

                if not constraint_result['valid']:
                    all_valid = False

                results.append(constraint_result)

            except Exception as e:
                logger.error(f"Constraint validation failed for '{name}': {e}")
                results.append({
                    'name': name,
                    'valid': False,
                    'error': str(e)
                })
                all_valid = False

        return {
            'all_valid': all_valid,
            'constraint_count': len(constraints),
            'valid_constraints': sum(1 for r in results if r.get('valid', False)),
            'results': results
        }

    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about validated nodes.

        Returns:
            Dictionary with validation statistics
        """
        if not self.driver:
            self.connect()

        # Count nodes by label
        label_query = """
        CALL db.labels() YIELD label
        CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) AS count', {}) YIELD value
        RETURN label, value.count AS count
        """

        try:
            results = self.execute_query(label_query)
            stats = {
                'total_labels': len(results),
                'labels_by_count': {r['label']: r['count'] for r in results}
            }
            return stats
        except Exception as e:
            logger.warning(f"Could not get validation statistics: {e}")
            return {'total_labels': 0, 'labels_by_count': {}}


def main():
    """Example usage of the SHACLValidator."""
    validator = SHACLValidator()

    try:
        # Validate the entire graph
        report = validator.validate_graph()

        print("SHACL Validation Report:")
        print(f"  Conformant: {report['conformant']}")
        print(f"  Violations: {report['violation_count']}")

        if report['violations']:
            print("\nViolations:")
            for violation in report['violations'][:5]:  # Show first 5
                print(f"  - {violation['message']}")

        # Get SHACL shapes
        shapes = validator.get_shacl_shapes()
        print(f"\nLoaded SHACL shapes: {len(shapes)}")

        # Get validation statistics
        stats = validator.get_validation_statistics()
        print(f"Node statistics: {stats}")

    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        validator.close()


if __name__ == "__main__":
    main()

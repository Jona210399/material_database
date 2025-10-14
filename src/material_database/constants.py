from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ColumnNames:
    ID: str = "id"
    SYMMETRIZED_STRUCTURE: str = "symmetrized_structure"
    CIF: str = "cif"

    PEAK_TWO_THETAS: str = "peak_two_thetas"
    PEAK_INTENSITIES: str = "peak_intensities"
    INTENSITIES: str = "intensities"

    ROBOCRYS: str = "robocrys"

    MATBIND_GRAPH_NODE_FEATURES: str = "node_features"
    MATBIND_GRAPH_EDGE_INDEX: str = "edge_index"
    MATBIND_GRAPH_EDGE_ATTRS: str = "edge_attrs"

    @classmethod
    def all_columns(cls):
        return [getattr(cls, field.name) for field in fields(cls)]

"""Algorithm registry: add new recommenders here to make them usable everywhere."""
from src.recommenders.popularity import PopularityRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.item_similarity import ItemSimilarityRecommender
from src.recommenders.collaborative import CollaborativeRecommender
from src.recommenders.association import AssociationRecommender

RECOMMENDER_CLASSES = {
    cls.name: cls for cls in (
        PopularityRecommender,
        ContentBasedRecommender,
        ItemSimilarityRecommender,
        CollaborativeRecommender,
        AssociationRecommender,
    )
}

__all__ = ["RECOMMENDER_CLASSES", "PopularityRecommender", "ContentBasedRecommender",
           "ItemSimilarityRecommender", "CollaborativeRecommender", "AssociationRecommender"]

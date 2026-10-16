import torch
import torch.nn as nn
import torch.nn.functional as nnF
from collections.abc import Sequence
from transformers.modeling_outputs import SequenceClassifierOutput

EMBED_DIM_L = 256
EMBED_DIM_M = 64
EMBED_DIM_S = 16

def create_student_model(name: str, vocab_size: int) -> nn.Module:
    """Create a configured student architecture by its saved model name."""
    configurations = {
        "MLP_DeepNarrow_L": (customMLP_wEmbedding, {"embed_dim": EMBED_DIM_L, "layers_array": [512] * 8}),
        "MLP_DeepNarrow_M": (customMLP_wEmbedding, {"embed_dim": EMBED_DIM_M, "layers_array": [256] * 6}),
        "MLP_DeepNarrow_S": (customMLP_wEmbedding, {"embed_dim": EMBED_DIM_S, "layers_array": [128] * 4}),
        "MLP_ShortWide_L": (customMLP_wEmbedding, {"embed_dim": EMBED_DIM_L, "layers_array": [4096, 4096]}),
        "MLP_ShortWide_M": (customMLP_wEmbedding, {"embed_dim": EMBED_DIM_M, "layers_array": [1024, 1024]}),
        "MLP_ShortWide_S": (customMLP_wEmbedding, {"embed_dim": EMBED_DIM_S, "layers_array": [512]}),
        "MLP_Bottleneck_L": (customMLP_wEmbedding, {"embed_dim": EMBED_DIM_L, "layers_array": [2048, 512, 2048]}),
        "MLP_Bottleneck_M": (customMLP_wEmbedding, {"embed_dim": EMBED_DIM_M, "layers_array": [512, 64, 512]}),
        "MLP_Bottleneck_S": (customMLP_wEmbedding, {"embed_dim": EMBED_DIM_S, "layers_array": [256, 32, 256]}),
        "Trans_DeepNarrow_L": (customTransformer, {"embed_dim": EMBED_DIM_L, "num_layers": 12, "d_model": 256, "dff": 512, "num_heads": 4}),
        "Trans_DeepNarrow_M": (customTransformer, {"embed_dim": EMBED_DIM_M, "num_layers": 6, "d_model": 128, "dff": 256, "num_heads": 2}),
        "Trans_DeepNarrow_S": (customTransformer, {"embed_dim": EMBED_DIM_S, "num_layers": 4, "d_model": 32, "dff": 64, "num_heads": 2}),
        "Trans_ShortWide_L": (customTransformer, {"embed_dim": EMBED_DIM_L, "num_layers": 2, "d_model": 1024, "dff": 2048, "num_heads": 8}),
        "Trans_ShortWide_M": (customTransformer, {"embed_dim": EMBED_DIM_M, "num_layers": 1, "d_model": 256, "dff": 512, "num_heads": 4}),
        "Trans_ShortWide_S": (customTransformer, {"embed_dim": EMBED_DIM_S, "num_layers": 1, "d_model": 64, "dff": 128, "num_heads": 2}),
        "Trans_Bottleneck_L": (customTransformerBN, {"embed_dim": EMBED_DIM_L, "d_model": [256, 64, 1024], "dff": [512, 128, 512], "num_heads": [4, 4, 4]}),
        "Trans_Bottleneck_M": (customTransformerBN, {"embed_dim": EMBED_DIM_M, "d_model": [128, 32, 128], "dff": [256, 64, 256], "num_heads": [2, 2, 2]}),
        "Trans_Bottleneck_S": (customTransformerBN, {"embed_dim": EMBED_DIM_S, "d_model": [32, 8, 32], "dff": [64, 16, 64], "num_heads": [2, 2, 2]}),
    }
    try:
        model_class, parameters = configurations[name]
    except KeyError as error:
        raise ValueError(f"Unknown student model name: {name}") from error
    return model_class(vocab_size=vocab_size, num_labels=6, name=name, **parameters)

class customMLP_wEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_labels: int = 6, layers_array: Sequence[int] = (), name: str = "customMLP"):
        super().__init__()
        self.name = name

        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        hidden_layers = []
        
        previous_size = embed_dim                               #First Layer is always the embedding
        for layer_size in layers_array:                         #Then for each layer
            hidden_layers.extend((
                nn.Linear(previous_size, layer_size),           #Simply linear connected to the prior layer
                nn.ReLU(),                                      #With relu activation function
            ))
            previous_size = layer_size

        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.classifier = nn.Linear(previous_size, num_labels)  #End with a linear layer to the number of labels

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = self.embeddings(input_ids)

        if attention_mask is None: 
            attention_mask = torch.ones_like(input_ids)

        m = attention_mask.unsqueeze(-1).to(x.dtype)    #Connection from embedding to the MLP
        x = (x * m).sum(1) / m.sum(1).clamp(min=1)      #No padding when batch=1 as in production
                   
        x = self.hidden_layers(x)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = nnF.binary_cross_entropy_with_logits(logits, labels.float())
        return SequenceClassifierOutput(logits=logits, loss=loss)


class customTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int = 6, d_model: int = 512, dff: int = 2048,
                 num_heads: int = 8, num_labels: int = 6, max_position_embeddings: int = 512, name: str = "customTransformer"):
        super().__init__()
        self.name = name

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.input_projection = (nn.Identity() if embed_dim == d_model else nn.Linear(embed_dim, d_model))
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)

        encoder_layer = nn.TransformerEncoderLayer(         #Build one transformer layer
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dff,
            batch_first=True,
        )
        self.transformer_model = nn.TransformerEncoder(     #Stack the layers
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(d_model, num_labels)    #Output layer

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        sequence_length = input_ids.size(1)

        if sequence_length > self.position_embeddings.num_embeddings:
            raise ValueError(
                f"Sequence length ({sequence_length}) exceeds the configured "
                f"maximum ({self.position_embeddings.num_embeddings})."
            )

        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        x = self.embeddings(input_ids)
        x = self.input_projection(x) + self.position_embeddings(positions)      #Input to transformer is output of embed + positional info

        padding_mask = None
        if attention_mask is not None:
            padding_mask = attention_mask == 0

        hidden_state = self.transformer_model(x, src_key_padding_mask=padding_mask)
        pooled_output = hidden_state[:, 0]  # Use the [CLS] token representation.
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nnF.binary_cross_entropy_with_logits(logits, labels.float())

        return SequenceClassifierOutput(logits=logits, loss=loss)



class customTransformerBN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, d_model: Sequence[int], dff: Sequence[int], 
                 num_heads: Sequence[int], num_labels: int = 6, max_position_embeddings: int = 512, name: str = "customTransformerBN"):
        super().__init__()
        self.name = name

        #Sanity checks
        if not d_model:
            raise ValueError("d_model must contain at least one transformer width.")
        if not (len(d_model) == len(dff) == len(num_heads)):
            raise ValueError(
                "d_model, dff, and num_heads must have the same length."
            )
        for width, heads in zip(d_model, num_heads):
            if width <= 0 or heads <= 0 or width % heads != 0:
                raise ValueError(
                    "Each transformer width must be positive and divisible "
                    "by its number of heads."
                )

        #Build the model
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.input_projection = (nn.Identity() if embed_dim == d_model[0] else nn.Linear(embed_dim, d_model[0]))
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model[0])

        self.transformer_layers = nn.ModuleList()                       #List of layers
        self.bottleneck_projections = nn.ModuleList()                   #Projections between layers

        for layer_idx, (width, feedforward, heads) in enumerate(zip(d_model, dff, num_heads)):  #Collect all inputs
            self.transformer_layers.append(nn.TransformerEncoderLayer(                          #Build individual layer
                                            d_model=width,
                                            nhead=heads,
                                            dim_feedforward=feedforward,
                                            batch_first=True,
                                            ))
            
            if layer_idx < len(d_model) - 1:
                self.bottleneck_projections.append(nn.Linear(width, d_model[layer_idx + 1]))    #Projection to next layer

        self.classifier = nn.Linear(d_model[-1], num_labels)                                    #Output layer

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        sequence_length = input_ids.size(1)

        if sequence_length > self.position_embeddings.num_embeddings:
            raise ValueError(
                f"Sequence length ({sequence_length}) exceeds the configured "
                f"maximum ({self.position_embeddings.num_embeddings})."
            )

        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        x = self.embeddings(input_ids)
        x = self.input_projection(x) + self.position_embeddings(positions)

        padding_mask = None
        if attention_mask is not None:
            padding_mask = attention_mask == 0

        for layer_idx, transformer_layer in enumerate(self.transformer_layers):     #Push through all layers
            x = transformer_layer(x, src_key_padding_mask=padding_mask)
            if layer_idx < len(self.bottleneck_projections):                        #If projection is needed
                x = self.bottleneck_projections[layer_idx](x)

        hidden_state = x
        pooled_output = hidden_state[:, 0]  # Use the [CLS] token representation.
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nnF.binary_cross_entropy_with_logits(logits, labels.float())

        return SequenceClassifierOutput(logits=logits, loss=loss)
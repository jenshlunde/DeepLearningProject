import torch
import torch.nn as nn
import torch.nn.functional as nnF
from collections.abc import Sequence
from transformers.modeling_outputs import SequenceClassifierOutput

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

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embeddings(input_ids)
        x = x.mean(dim=1)                       #Connection from embedding to the MLP
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
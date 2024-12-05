import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaEncoder,
    RobertaLayer,
    RobertaEmbeddings,
    RobertaConfig,
)

class CustomRobertaLayer(RobertaLayer):
    def __init__(self, config):
        super().__init__(config)
        self.down_layer = nn.Linear(config.hidden_size, config.hidden_size // 2)  # Down-projection
        self.up_layer = nn.Linear(config.hidden_size // 2, config.hidden_size)    # Up-projection
        self.activation = nn.GELU()                                              # Activation function
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(config.hidden_size)                       # LayerNorm after Up-projection
        # intializing all as new layers
        self.down_layer._is_new = True
        self.up_layer._is_new = True
        self.activation._is_new = True
        self.layer_norm._is_new = True

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        # Ensure the attention mask matches the required dimensions
        if attention_mask is not None:
            # Expand dimensions for multi-head attention
            attention_mask = attention_mask[:, None, None, :]  # Shape: [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # Match precision (e.g., float16)


        # Attention sub-layer
        attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0]

        # Feed-forward sub-layer
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(hidden_states=intermediate_output, input_tensor=attention_output)

        # Down-projection, activation, up-projection, and LayerNorm
        down_projected = self.activation(self.down_layer(layer_output))
        down_projected = self.dropout(down_projected)
        up_projected = self.activation(self.up_layer(down_projected))
        norm_output = self.layer_norm(up_projected + layer_output)  # Residual connection

        return (norm_output,) + attention_outputs[1:]  # Return outputs

# Custom Encoder
class CustomRobertaEncoder(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([CustomRobertaLayer(config) for _ in range(config.num_hidden_layers)])

# Custom Model
class CustomRobertaModel(RobertaModel):
    def __init__(self, config):
        super().__init__(config)

        # Replace the encoder with the custom encoder
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = CustomRobertaEncoder(config)

        # Add the classification head at the end
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, config.num_labels),
        )
        self.classifier._is_new = True

        # Freeze existing layers if needed
        self.freeze_pretrained_layers()

    def freeze_pretrained_layers(self):
        # Freeze all layers except the classifier and custom layers
        for name, param in self.named_parameters():
            if "classifier" in name or "down_layer" in name or "up_layer" in name or "layer_norm" in name or "attention" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
    ):
        # Embedding layer
        embedding_output = self.embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)

        # Encoder layer
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]

        # Extract the [CLS] token representation
        cls_token_output = sequence_output[:, 0, :]

       # Classification head
        logits = self.classifier(cls_token_output)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        # Return loss if available, otherwise logits
        return (loss, logits) if loss is not None else logits

# Initialize new weights
def initialize_weights(module):
    if isinstance(module, nn.Linear) and getattr(module, "_is_new", False):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
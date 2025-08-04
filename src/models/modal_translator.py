import torch
from torch import nn
from torch import Tensor


class ModalTranslator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        in_seq: int,
        out_seq: int,
        num_heads: int,
        torch_dtype: torch.dtype,
        overlap_inputs: bool,
    ):  # in_seq should always be the same
        if input_dim % num_heads != 0:
            raise ValueError(
                f"InputDim {input_dim} is not devidable by num_heads {num_heads}."
            )

        if in_seq % 2 != 0 and overlap_inputs:
            print(
                f"WARNING: in_seq {in_seq} is not even and overlap is enabled.\n Some input sequences may be lost."
            )

        super().__init__()

        self.in_seq = in_seq
        self.out_seq = out_seq
        self.torch_dtype = torch_dtype
        self.overlap_inputs = overlap_inputs

        self.query = nn.Parameter(torch.randn(out_seq, input_dim, dtype=torch_dtype))
        self.attention = nn.MultiheadAttention(
            input_dim,
            num_heads,
            batch_first=True,
            dtype=torch_dtype,
        )
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim, dtype=torch_dtype),
            nn.GELU(),
            nn.Linear(output_dim, output_dim, dtype=torch_dtype),
            nn.GELU(),
            nn.Linear(output_dim, output_dim, dtype=torch_dtype),
        )
        self.audio_embedding = nn.Parameter(torch.randn(output_dim, dtype=torch_dtype))

    def chunk(
        self,
        audio_embed: Tensor,  # [batch, seq, hidden]
        key_mask: Tensor,  # [batch, seq]
    ) -> tuple[Tensor, Tensor]:

        if self.overlap_inputs:
            # fmt: off
            embed_chunks = audio_embed.unfold(1, self.in_seq, self.in_seq // 2) # [batch, chunks, hidden, in_seq]
            mask_chunks = key_mask.unfold(1, self.in_seq, self.in_seq // 2) # [batch, chunks, in_seq]

            embed_chunks = embed_chunks.permute(0, 1, 3, 2)  # [batch, chunks, in_seq, hidden]
            # fmt: on

        else:
            shape = audio_embed.size()
            reshape = (
                shape[0],  # batch
                -1,  # chunks
                self.in_seq,  # in_seq
            )

            embed_chunks = audio_embed.view(
                *reshape,
                shape[2],  # hidden
            )
            mask_chunks = key_mask.view(reshape)

        return embed_chunks, mask_chunks

    def prevent_none(
        self,
        mask_chunks: Tensor,  # [batch * chunks, in_seq]
    ) -> Tensor:
        first_unmasked = torch.zeros_like(mask_chunks)
        first_unmasked[:, 0] = 1
        empty_chunks = (mask_chunks == 0).all(dim=1, keepdim=True)
        return mask_chunks + first_unmasked * empty_chunks

    def forward(
        self,
        audio_embed: Tensor,  # [batch, seq, hidden]
        key_mask: Tensor,  # [batch, seq]
    ) -> tuple[Tensor, Tensor]:
        if audio_embed.size(1) % self.in_seq != 0:
            raise ValueError(
                f"Input seq lenght {audio_embed.size(1)} is not devidable by {self.in_seq}."
            )

        embed_chunks, mask_chunks = self.chunk(
            audio_embed, key_mask
        )  # [batch, chunks, in_seq, (hidden)]

        shape = embed_chunks.size()
        embed_chunks = embed_chunks.reshape(
            -1,  # batch * chunks
            shape[2],  # in_seq
            shape[3],  # hidden
        )
        mask_chunks = mask_chunks.reshape(
            -1,  # batch * chunks
            shape[2],  # in_seq
        )

        queries = self.query.unsqueeze(0).expand(embed_chunks.size(0), -1, -1)

        temp_mask = self.prevent_none(mask_chunks)
        compressed_chunks: Tensor = self.attention(  # [batch * chunks, out_seq, hidden]
            queries,
            embed_chunks,
            embed_chunks,
            key_padding_mask=(temp_mask == 0),
        )[0]

        compressed_chunks = compressed_chunks.reshape(
            shape[0],  # batch
            -1,  # chunks * out_seq
            shape[3],  # hidden
        )

        translated = self.projection(compressed_chunks)

        mask_chunks = mask_chunks.view(
            shape[0],  # batch
            shape[1],  # chunks
            shape[2],  # in_seq
        )
        entirely_padding = (mask_chunks == 0).all(dim=2, keepdim=True).logical_not()

        translated_mask = entirely_padding.repeat(
            1, 1, self.out_seq
        )  # [batch, chunks, out_seq]
        translated_mask = translated_mask.view(
            shape[0],  # batch
            -1,  # chunks * out_seq
        )

        return (
            translated + self.audio_embedding,  # [batch, chunks * out_seq, output_dim]
            translated_mask.to(torch.int32),  # [batch, chunks * out_seq]
        )

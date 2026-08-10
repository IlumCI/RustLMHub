Published `config.json` files, copied verbatim from the Hub, one per architecture the
port targets. They are fixtures rather than a download step because the descriptor in
`rust/src/arch.rs` is only worth anything if it parses the files the models actually ship
-- a hand-written approximation would agree with itself and with nothing else.

  deepseek_v4_flash.json  deepseek-ai/DeepSeek-V4-Flash-0731
  glm_5_1.json            zai-org/GLM-5.1
  minimax_m3.json         MiniMaxAI/MiniMax-M3

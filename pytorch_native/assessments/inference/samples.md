# 推理生成样例

参数：现有训练检查点、原始 generate()、BF16、temperature=0.8、top_p=0.9、seed=42、最多 128 个新 token。每例同 seed 重复生成的 token 序列一致。以下四例仅作定性观察。

## 样例 1

Prompt：Once upon a time

输入 4 token；生成 128 token；特殊 token 早停：False。重复请求耗时 1.151 秒，首 token 7.53 ms。

```text
Once upon a time, there was a big, impressive tree. It had many green leaves and tall branches. One day, a little girl named Lily went to the tree. She wanted to see the tree.
As she got closer, she saw a squirrel. The squirrel was eating a nut. Lily was sad. She wanted to help the squirrel. She looked at the tree and saw the nut. Lily thought, "I will give the nut to the squirrel."
Lily gave the nut to the squirrel. The squirrel ate the nut and was happy. But then, something unexpected happened. The squirrel started to grow bigger and bigger! It was not a
```

## 样例 2

Prompt：Lily found a small red box in the garden. Inside the box

输入 13 token；生成 89 token；特殊 token 早停：True。重复请求耗时 0.805 秒，首 token 7.38 ms。

```text
Lily found a small red box in the garden. Inside the box, there was a shiny crown. Lily put the crown on her head and started to dance. She was so happy with her new crown.
But then, something unexpected happened. The crown started to talk! It said, "Hello, Lily! I am a magic crown. I can make your wishes come true!" Lily was surprised but excited. She wished for a big, fun playground. The magic crown made her wish come true.

```

## 样例 3

Prompt：Explain why the sky is blue.

输入 8 token；生成 1 token；特殊 token 早停：True。重复请求耗时 0.014 秒，首 token 7.07 ms。

```text
Explain why the sky is blue.

```

## 样例 4

Prompt：请介绍一下你自己。

输入 27 token；生成 92 token；特殊 token 早停：True。重复请求耗时 0.848 秒，首 token 7.75 ms。

```text
请介绍一下你自己。 was a happy one, and so on top of it was the worst idea of all. She quickly closed her eyes and imagined a magical device that would take her for a ride.
When she was finally close enough, the badger realised that the pink shampoo was actually a secret mission! It was a long journey and she was eager to start her adventure. She slowly opened her eyes, and started dreaming about what was inside the pink shampoo.

```

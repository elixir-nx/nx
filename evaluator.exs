Mix.install([
  {:bumblebee, "~> 0.6.0"},
  {:nx, path: "/Users/valente/coding/nx/nx", override: true},
  {:emlx, path: "/Users/valente/coding/emlx"},
])

Nx.global_default_backend({EMLX.Backend, device: :gpu})
Nx.Defn.default_options(compiler: EMLX)

defmodule MyDefn do
  import Nx.Defn

  defn add_mul(a, b) do
    x = a + b
    x * x
  end
end

serving = Nx.Serving.new(fn opts -> Nx.Defn.jit(&MyDefn.add_mul/2) end
  Bumblebee.Diffusion.StableDiffusion.text_to_image(clip, unet, vae, tokenizer, scheduler,
    num_steps: 20,
    num_images_per_prompt: 1,
    safety_checker: safety_checker,
    safety_checker_featurizer: featurizer,
    compile: [batch_size: 1, sequence_length: 60],
    # Option 1
    # defn_options: []
    # Option 2 (reduces GPU usage, but runs noticeably slower)
    # Also remove `backend: EMLX.Backend` from the loading options above
    # defn_options: [compiler: EXLA, lazy_transfers: :always]
  )

Nx.Serving.start_link(name: StableDiffusion, serving: serving)

prompt = "numbat, forest, high quality, detailed, digital art"
negative_prompt = "darkness, rainy, foggy"

IO.puts("Running batched_run")
:timer.tc(fn ->
  Nx.Serving.batched_run(StableDiffusion, %{prompt: prompt, negative_prompt: negative_prompt})
end)
|> IO.inspect(label: "batched_run")

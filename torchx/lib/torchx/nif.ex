defmodule Torchx.NIF.Macro do
  @moduledoc false

  defmacro dnif(call) do
    {name, args} = Macro.decompose_call(call)
    name_io = :"#{name}_io"
    args = underscore_args(args)

    quote do
      def unquote(name)(unquote_splicing(args)), do: :erlang.nif_error(:undef)
      def unquote(name_io)(unquote_splicing(args)), do: :erlang.nif_error(:undef)
    end
  end

  defp underscore_args(args) do
    Enum.map(args, fn {name, meta, args_list} -> {:"_#{name}", meta, args_list} end)
  end
end

defmodule Torchx.NIF do
  import Torchx.NIF.Macro

  @moduledoc false
  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:torchx), 'torchx')
    :erlang.load_nif(path, 0)
  end

  for {op, args} <- Torchx.__torch__() do
    def unquote(op)(unquote_splicing(Macro.generate_arguments(length(args), __MODULE__))),
      do: :erlang.nif_error(:undef)

    def unquote(:"#{op}_io")(
          unquote_splicing(Macro.generate_arguments(length(args), __MODULE__))
        ),
        do: :erlang.nif_error(:undef)
  end

  dnif tensordot(tensorA, tensorB, axesA, axesB)
  dnif matmul(tensorA, tensorB)

  dnif cuda_is_available()
  dnif cuda_device_count()

  def item(_tensor), do: :erlang.nif_error(:undef)
  def scalar_type(_tensor), do: :erlang.nif_error(:undef)
  def shape(_tensor), do: :erlang.nif_error(:undef)
  def names(_tensor), do: :erlang.nif_error(:undef)
  def strides(_tensor), do: :erlang.nif_error(:undef)
  def device_of(_tensor), do: :erlang.nif_error(:undef)
  def nbytes(_tensor), do: :erlang.nif_error(:undef)
  def to_blob_view(_tensor), do: :erlang.nif_error(:undef)

  def call(func, :cpu, args) when is_atom(func) and is_list(args),
    do: apply(__MODULE__, func, args |> convert_device_arg(:cpu))

  def call(func, device, args) when is_atom(func) and is_list(args),
    do: apply(__MODULE__, :"#{func}_io", args |> convert_device_arg(device))

  defp convert_device_arg(args, device),
    do:
      Enum.map(
        args,
        fn
          ^device -> Torchx.torch_device(device)
          arg -> arg
        end
      )
end

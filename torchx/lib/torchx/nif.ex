defmodule Torchx.NIF.Macro do
  defmacro dnif(call) do
    {name, args} = Macro.decompose_call(call)
    name_io = :"#{name}_io"
    args = underscore_args(args)

    quote do
      def unquote(name)(unquote_splicing(args)), do: :erlang.nif_error(:undef)
      def unquote(name_io)(unquote_splicing(args)), do: :erlang.nif_error(:undef)
    end
  end

  defp underscore_args(args),
    do:
      args
      |> Enum.map(fn {name, meta, args_list} -> {:"_#{name}", meta, args_list} end)
end

defmodule Torchx.NIF do
  import Torchx.NIF.Macro

  @moduledoc false
  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:torchx), 'torchx')
    :erlang.load_nif(path, 0)
  end

  dnif(randint(min, max, shape, type))
  dnif(rand(min, max, shape, type))

  dnif(scalar_tensor(scalar, type))
  dnif(normal(mu, sigma, shape))
  dnif(arange(from, to, step, type))
  dnif(arange(from, to, step, type, shape))
  dnif(iota(shape, axis, type))
  dnif(reshape(tensor, shape))
  dnif(to_type(tensor, type))
  dnif(from_blob(blob, shape, type))
  dnif(to_blob(tensor))
  dnif(to_blob(tensor, limit))

  dnif(delete_tensor(tensor))
  dnif(squeeze(tensor))
  dnif(squeeze(tensor, axis))
  dnif(broadcast_to(tensor, shape))
  dnif(transpose(tensor, dim0, dim1))
  dnif(split(tensor, split_size))

  dnif(ones(shape))
  dnif(eye(size, type))
  dnif(add(tensorA, tensorB))
  dnif(dot(tensorA, tensorB))
  dnif(cholesky(tensor))
  dnif(cholesky(tensor, upper))

  dnif(qr(tensor))
  dnif(qr(tensor, reduced))

  def type(_tensor), do: :erlang.nif_error(:undef)
  def device(_tensor), do: :erlang.nif_error(:undef)
  def nbytes(_tensor), do: :erlang.nif_error(:undef)
end

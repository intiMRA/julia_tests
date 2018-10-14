#imports minist dataset
using Flux, Flux.Data.MNIST, Statistics
#RESEARCH
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
# using CuArrays

# Classify MNIST digits with a convolutional network

print("Start\n")
#gets all images
imgs = MNIST.images()
print("IMAGES\n")

labels = onehotbatch(MNIST.labels(), 0:9)
print("TRAIN\n")

# Partition into batches of size 1,000
train = [(cat(float.(imgs[i])..., dims = 4), labels[:,i])
         for i in partition(1:60_000, 1000)]

print("FINISH Creating TRAIN\n")
#generates train
train = gpu.(train)

# Prepare test set (first 1,000 images)
print("CREATE TEST\n")
tX = cat(float.(MNIST.images(:test)[1:1000])..., dims = 4) |> gpu
tY = onehotbatch(MNIST.labels(:test)[1:1000], 0:9) |> gpu

#creates model 2 convo layers 2,2 and 1 fully conectet 288,10
print("Create Model\n")
m = Chain(
  Conv((2,2), 1=>16, relu),
  x -> maxpool(x, (2,2)),
  Conv((2,2), 16=>8, relu),
  x -> maxpool(x, (2,2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(288, 10), softmax) |> gpu

#passes data into model
m(train[1][1])

#loss function
loss(x, y) = crossentropy(m(x), y)

#mesure of accuracy
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))
#plots?
evalcb = throttle(() -> @show(accuracy(tX, tY)), 10)
opt = ADAM(params(m))
#trins
print("FIT\n")
for i=1:10
  println("epoch: ",i)
  Flux.train!(loss, train, opt, cb = evalcb)
end

local SingleBatcher, parent = torch.class('SingleBatcher')
    
function SingleBatcher:__init(fileList,batchsize,onepass,wholeFileCuda,preprocessCmd)
    self.preprocessCmd = preprocessCmd
    self.wholeFileCuda = wholeFileCuda 
    self.dataFileIndex = 1
    self.dataIndex = 1
    self.onepass = onepass
    self.mb = batchsize
    print("reading pre-cached data from: "..fileList)
    self.dataFiles = self:readList(fileList) 
    self.numProcessedFromFile = 0
    

end

function SingleBatcher:preprocess(l,i,n)
    if(self.preprocessCmd) then 
        return self.preprocessCmd(l,i,n)
    else
        return l,i,n
    end
end

function SingleBatcher:toCuda(s)
    return Util:deep_apply(s,function(s) return s:cuda() end)
end

function SingleBatcher:reset()
    self.dataIndex = 1
    self.numProcessedFromFile = 0
end

function SingleBatcher:loadNextBatch()
        if(#self.dataFiles > 1 or not self.loadedData) then 
            local fn = self.dataFiles[self.dataFileIndex]
            print('loading from: '..fn)
            self.loadedData = torch.load(fn)
            if(self.wholeFileCuda) then self.loadedData = self:toCuda(self.loadedData) end
        else
            local inds = torch.randperm(self.loadedData[1]:size(1)):long()
            self.loadedData = Util:deep_apply(self.loadedData,function(t) return t:index(1,inds) end)
        end
end
function SingleBatcher:getBatch()
    if(not self.loadedData) then
        self:loadNextBatch()
    end

    local endfile = false
    if(self.dataIndex > self.loadedData[1]:size(1)) then 
        endfile = true
        assert(self.numProcessedFromFile == self.loadedData[1]:size(1),"numProcessed = "..self.numProcessedFromFile.." size = "..self.loadedData[1]:size(1))
        --self.loadedData = nil 
        self.dataFileIndex = self.dataFileIndex + 1
        if(self.dataFileIndex > #self.dataFiles) then self.dataFileIndex = 1 end
        local dataFile = self.dataFiles[self.dataFileIndex]
        self:loadNextBatch()
        self.numProcessedFromFile = 0
        self.dataIndex = 1
        if(self.onepass)then 
            --print('SingleBatcher finished processing '..self.numProcessedFromFile.." examples")
            return nil, nil, endfile 
        end
    end

    local len = (self.dataIndex + self.mb -1 <= self.loadedData[1]:size(1)) and self.mb or (self.loadedData[1]:size(1) - self.dataIndex + 1)
    local iptr = self.loadedData[2]:narrow(1,self.dataIndex,len)
    local lptr = self.loadedData[1]:narrow(1,self.dataIndex,len)
    
    self.numProcessedFromFile = self.numProcessedFromFile + len
    self.dataIndex = self.dataIndex + self.mb


    if(not self.wholeFileCuda) then
        self.cudaInput = self.cudaInput or iptr:cuda()
        self.cudaLabel = self.cudaLabel or lptr:cuda()
        self.cudaInput:resize(iptr:size()):copy(iptr)
        self.cudaLabel:resize(lptr:size()):copy(lptr)
        return self.cudaLabel, self.cudaInput, len
    else
        return self:preprocess(lptr, iptr, len)
    end


end

function SingleBatcher:readList(file)
    local tab = {}
    for l in io.lines(file) do
        table.insert(tab,l)
    end
    return tab
end


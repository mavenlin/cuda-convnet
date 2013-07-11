% generate unit test for img_acts
filter_acts = 1;
imgSize = 10;
imgNum  = 10;
actSize = 6;
filterSize = 5;
filterNum = 4;
imgColor = 2;

randImg = rand(imgColor*imgSize^2, imgNum);
randFilter = rand(imgColor*filterSize^2, filterNum);
randActs = rand(filterNum*actSize^2, imgNum);


Imgs = {};
for i = 1:imgNum
    imgs = [];
    for j = 1:imgColor
        imgs(:,:,j) = reshape(randImg(((j-1)*imgSize^2+1):j*imgSize^2, i), imgSize, imgSize);
    end
    Imgs{i} = imgs;
end

Filters = {};
for i = 1:filterNum
    filters = [];
    for j = 1:imgColor
        filters(:,:,j) = reshape(randFilter(((j-1)*filterSize^2+1):j*filterSize^2, i),filterSize, filterSize);
    end
    Filters{i} = filters;
end

Acts = {};
for i = 1:imgNum
    acts = [];
    for j = 1:filterNum
        acts(:,:,j) = reshape(randActs(((j-1)*actSize^2+1):j*actSize^2, i), actSize, actSize);
    end
    Acts{i} = acts;
end

resultact=[];
if filter_acts
    for i = 1:imgNum
        curract = zeros(actSize, actSize, filterNum);
        for k = 1:imgColor
            for j = 1:filterNum
                filter = Filters{j};
                img = Imgs{i};
                curract(:,:,j) = curract(:,:,j) + convn(img(:,:,k), fliplr(flipud(filter(:,:,k))), 'valid');
            end
        end
        resultact = [resultact, curract(:)];
    end
    randImg = randImg';
    randImg = randImg(:)';
    randFilter = randFilter';
    randFilter = randFilter(:)';
    resultact = resultact';
    resultact = resultact(:)';
end

out = [imgSize, imgNum, imgColor, filterSize, filterNum, actSize, randImg, randFilter, resultact];
dlmwrite('test/filter_acts', out, 'precision', '%.20f', 'delimiter', ' ');


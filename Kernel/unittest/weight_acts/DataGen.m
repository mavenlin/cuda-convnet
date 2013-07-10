% generate unit test for img_acts
weight_acts = 1;
imgSize = 10;
imgNum  = 10;
actSize = 6;
filterSize = 5;
filterNum = 2;
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

resultFilter = [];
if weight_acts
    for i = 1:filterNum
        currFilter = zeros(filterSize, filterSize, imgColor);
        for k = 1:imgColor
            for j = 1:imgNum
                img = Imgs{j};
                act = Acts{j};
                currFilter(:,:,k) = currFilter(:,:,k) + convn(img(:,:,k), fliplr(flipud(act(:,:,i))), 'valid');
            end
        end
        resultFilter = [resultFilter, currFilter(:)];
    end
    randImg = randImg';
    randImg = randImg(:)';
    resultFilter = resultFilter';
    resultFilter = resultFilter(:)';
    randActs = randActs';
    randActs = randActs(:)';
end

out = [imgSize, imgNum, imgColor, filterSize, filterNum, actSize, randImg, resultFilter, randActs];
dlmwrite('weight_acts', out, 'precision', '%.20f', 'delimiter', ' ');


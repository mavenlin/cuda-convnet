channels = 3;
cases = 12;
imgSize = 16;
labelnum = 10;

data=rand(imgSize*imgSize*channels, cases);
labels = randi(labelnum, 1, cases);
sq = data.^2;
compact = zeros(channels, cases);

for i = 1:cases
    for j = 1:channels
        compact(j, i) = compact(j, i) + sum(sq((j-1)*imgSize*imgSize+1:imgSize*imgSize*j, i));
    end
end

[a,b,c] = unique(labels);
comcompact = zeros(channels, numel(a));
for i = 1:cases
    comcompact(:,c(i)) = comcompact(:,c(i)) + compact(:,i);
end

for i = 1:numel(a)
    comcompact(:, i) = comcompact(:, i) * sum((labels==a(i)));
end
comcompact = sqrt(comcompact);

cost = sum(sum(comcompact));

gradient = zeros(size(data));

for i = 1:cases
    for j = 1:channels
        index = find(a==labels(i));
        gradient((j-1)*imgSize*imgSize+1:imgSize*imgSize*j, i) = sum(labels == labels(i)) * data((j-1)*imgSize*imgSize+1:imgSize*imgSize*j, i) ./ (comcompact(j, index));
    end
end

dd = data';
dd = dd(:)';

gg = gradient';
gg = gg(:)';

out = [imgSize, cases, channels, labels, dd, gg, cost];
dlmwrite('groupsparse', out, 'precision', '%.20f', 'delimiter', ' ');
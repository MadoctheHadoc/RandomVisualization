import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class MultiPivotQuickSort {

    public static void quickSort(int[] arr, int numPivots) {
        if (numPivots < 1)
            throw new IllegalArgumentException("Number of pivots must be >= 1");
        quickSortRecursive(arr, 0, arr.length - 1, numPivots);
    }

    private static void quickSortRecursive(int[] arr, int low, int high, int numPivots) {
        if (low >= high)
            return;

        int length = high - low + 1;
        
        // Use insertion sort for small subarrays
        if (length <= 10) {
            insertionSort(arr, low, high);
            return;
        }

        // Adjust numPivots if range is too small
        int actualPivots = Math.min(numPivots, length / 2);
        if (actualPivots < 1) actualPivots = 1;
        
        // Choose actualPivots distinct random pivots
        Random rand = new Random();
        Set<Integer> pivotIndices = new HashSet<>();
        while (pivotIndices.size() < actualPivots) {
            pivotIndices.add(low + rand.nextInt(length));
        }

        int[] pivotValues = new int[actualPivots];
        int i = 0;
        for (int idx : pivotIndices)
            pivotValues[i++] = arr[idx];

        Arrays.sort(pivotValues);

        // Partition using simple two-pass approach
        int[] partitionEnds = simplePartition(arr, low, high, pivotValues);

        // Recursively sort each partition (excluding pivot values)
        int start = low;
        for (int j = 0; j < partitionEnds.length; j++) {
            int end = partitionEnds[j];
            if (start < end) {
                quickSortRecursive(arr, start, end - 1, numPivots);
            }
            start = end;
        }
        
        // Sort last partition
        if (start <= high) {
            quickSortRecursive(arr, start, high, numPivots);
        }
    }

    // Two-pass partition approach
    private static int[] simplePartition(int[] arr, int low, int high, int[] pivots) {
        int k = pivots.length;
        int[] boundaries = new int[k];
        
        // Create temporary array to hold rearranged elements
        int[] temp = new int[high - low + 1];
        int[] counts = new int[k + 1];
        
        // Count elements in each partition
        for (int i = low; i <= high; i++) {
            int region = findRegion(arr[i], pivots);
            counts[region]++;
        }
        
        // Calculate boundary positions
        boundaries[0] = low + counts[0];
        for (int i = 1; i < k; i++) {
            boundaries[i] = boundaries[i-1] + counts[i];
        }
        
        // Calculate write positions for each partition
        int[] writePos = new int[k + 1];
        writePos[0] = low;
        for (int i = 1; i <= k; i++) {
            writePos[i] = boundaries[i-1];
        }
        
        // Distribute elements into temp array
        for (int i = low; i <= high; i++) {
            int value = arr[i];
            int region = findRegion(value, pivots);
            temp[writePos[region] - low] = value;
            writePos[region]++;
        }
        
        // Copy back to original array
        for (int i = 0; i < temp.length; i++) {
            arr[low + i] = temp[i];
        }
        
        return boundaries;
    }
    
    // Find which partition region a value belongs to
    // Returns 0 if value < pivots[0]
    // Returns i if pivots[i-1] <= value < pivots[i]
    // Returns k if value >= pivots[k-1]
    private static int findRegion(int value, int[] pivots) {
        int region = 0;
        while (region < pivots.length && value >= pivots[region]) {
            region++;
        }
        return region;
    }

    private static void insertionSort(int[] arr, int low, int high) {
        for (int i = low + 1; i <= high; i++) {
            int key = arr[i];
            int j = i - 1;
            while (j >= low && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }

    private static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    // Generate random array of given length with values from 0 to length-1
    public static int[] generateRandomArray(int length) {
        Random rand = new Random();
        int[] arr = new int[length];
        for (int i = 0; i < length; i++) {
            arr[i] = rand.nextInt(length);
        }
        return arr;
    }
    
    // Generate random array with custom seed for reproducibility
    public static int[] generateRandomArray(int length, long seed) {
        Random rand = new Random(seed);
        int[] arr = new int[length];
        for (int i = 0; i < length; i++) {
            arr[i] = rand.nextInt(length);
        }
        return arr;
    }

    public static boolean isSorted(int[] array) {
        for (int i = 0; i < array.length - 1; i++) {
            if (array[i] > array[i + 1]) {
                return false;
            }
        }
        return true;
    }

    public static void main(String[] args) {
        // Define array sizes (logarithmic scale)
        int[] sizes = {1, 2, 4, 8, 16, 32, 64};

        // Define different pivot counts to test
        int[] pivotCounts = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

        int repetitions = 10;

        String csvFile = "data/quicksort_results.csv";

        try (PrintWriter writer = new PrintWriter(new FileWriter(csvFile))) {
            // Write CSV header
            writer.println("ArraySize,Pivots,TimeMillis");

            System.out.printf("%-10s %-10s %-10s%n", "Size", "Pivots", "Time (ms)");
            System.out.println("--------------------------------");

            for (int size : sizes) {
                for (int pivots : pivotCounts) {
                    long start = System.nanoTime();
                    for (int ignored = 0; ignored < repetitions; ignored++) {
                        int[] x = generateRandomArray(size * 1000);
                        quickSort(x, pivots);
                    } 
                    long end = System.nanoTime();

                    long durationMs = ((end - start) / 1_000_000) * repetitions;

                    // // Verify correctness
                    // if (!isSorted(x)) {
                    //     System.out.println("⚠️ Array not sorted correctly for size " + size + " pivots " + pivots);
                    // }

                    // Print results to console
                    System.out.printf("%-10d %-10d %-10d%n", size, pivots, durationMs);

                    // Save results to CSV
                    writer.printf("%d,%d,%d%n", size, pivots, durationMs);
                }
            }

            System.out.println("Benchmark complete! Results saved to " + csvFile);
        } catch (IOException e) {
            System.err.println("Error writing CSV: " + e.getMessage());
        }
    }
}
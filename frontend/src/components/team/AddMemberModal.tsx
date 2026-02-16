import { Modal, Button } from "@/components/ui";

interface AddMemberModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAdd: (data: any) => void;
}

export default function AddMemberModal({
  isOpen,
  onClose,
  onAdd,
}: AddMemberModalProps) {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Handle form submission
    onAdd({});
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Add Team Member" size="2xl">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label
            className="block text-sm font-medium mb-2"
            style={{ color: "var(--text-secondary)" }}
          >
            Full Name
          </label>
          <input
            type="text"
            className="w-full px-4 py-3 rounded-lg placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
            style={{
              background: "#ffffff",
              border: "1px solid var(--border)",
              color: "var(--text-primary)",
            }}
            placeholder="John Doe"
          />
        </div>
        <div>
          <label
            className="block text-sm font-medium mb-2"
            style={{ color: "var(--text-secondary)" }}
          >
            Email
          </label>
          <input
            type="email"
            className="w-full px-4 py-3 rounded-lg placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
            style={{
              background: "#ffffff",
              border: "1px solid var(--border)",
              color: "var(--text-primary)",
            }}
            placeholder="john.doe@example.com"
          />
        </div>
        <div>
          <label
            className="block text-sm font-medium mb-2"
            style={{ color: "var(--text-secondary)" }}
          >
            Phone
          </label>
          <input
            type="tel"
            className="w-full px-4 py-3 rounded-lg placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
            style={{
              background: "#ffffff",
              border: "1px solid var(--border)",
              color: "var(--text-primary)",
            }}
            placeholder="+1 234 567 8900"
          />
        </div>
        <div className="flex gap-4 mt-6">
          <Button
            type="button"
            onClick={onClose}
            variant="secondary"
            className="flex-1"
          >
            Cancel
          </Button>
          <Button type="submit" variant="primary" className="flex-1">
            Add Member
          </Button>
        </div>
      </form>
    </Modal>
  );
}

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
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Full Name
          </label>
          <input
            type="text"
            className="w-full px-4 py-3 bg-slate-900/50 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500"
            placeholder="John Doe"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Email
          </label>
          <input
            type="email"
            className="w-full px-4 py-3 bg-slate-900/50 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500"
            placeholder="john.doe@example.com"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Phone
          </label>
          <input
            type="tel"
            className="w-full px-4 py-3 bg-slate-900/50 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500"
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
